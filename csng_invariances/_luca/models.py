#%%
import torch
from torch._C import default_generator
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import dnn_blocks as bl
import re
import utils
import wandb
from piqa import SSIM


class SSIMLoss(SSIM):
    def __init__(self):
        super().__init__(window_size=5, sigma=0.7, n_channels=1)

    def forward(self, x, y):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        y = y.reshape(-1, 1, y.shape[-2], y.shape[-1])
        return 1.0 - super().forward(x, y)


def loss_f(loss_name):
    if loss_name == "mse":
        loss = nn.MSELoss()
    if loss_name == "l1":
        loss = nn.L1Loss()
    if loss_name == "ssim":
        loss = SSIMLoss()
    return loss


class Base_ImgReconstructor(pl.LightningModule):
    """
    Base class for image reconstruction task
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = loss_f(config["loss"])
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.lr = config["lr"]
        self.reg = self.create_reg_dictionary()

    def forward(self, x):
        return x

    def create_reg_dictionary(self):
        keys = list(self.config.keys())
        r = re.compile("reg_.*")
        reg_keys = list(filter(r.match, keys))
        reg_dict = {}
        for key in reg_keys:
            if self.config[key] != 0:
                reg_dict[key[4:]] = self.config[key]
        return reg_dict

    def add_regularization(self, loss):
        if list(self.reg.keys()) == []:
            reg_loss = loss
        else:
            reg_terms = [getattr(self, reg)() for reg in self.reg.keys()]
            reg_loss = loss + torch.stack(reg_terms).sum()
        return reg_loss

    def training_step(self, train_batch, batch_idx):
        stim, resp = train_batch
        stim = stim.float()
        resp = resp.float()
        pred = self.forward(resp)
        loss = self.criterion(pred, stim)
        self.log("train/loss", loss)
        reg_loss = self.add_regularization(loss)
        self.log("train/loss_+_reg", reg_loss)
        return reg_loss

    def validation_step(self, val_batch, batch_idx):
        stim, resp = val_batch
        stim = stim.float()
        resp = resp.float()
        pred = self.forward(resp)
        loss = self.criterion(pred, stim)
        self.log("val/loss", loss)
        return {"val/loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val/loss"] for x in outputs]).mean()
        self.log("epoch_val_loss", avg_loss)

    def test_step(self, test_batch, batch_idx):
        stim, resp = test_batch
        stim = stim.float()
        resp = resp.float()
        pred = self.forward(resp)
        loss = self.criterion(pred, stim)
        self.log("test/loss", loss)
        return {"test/loss": loss}

    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    #     model_filename = "model_final.onnx"
    #     torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
    #     wandb.save(model_filename)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class Linear_ImgReconstructor(Base_ImgReconstructor):
    """
    Linear model for image reconstruction
    """

    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(self.input_size, self.output_size ** 2)

    def L1(self):
        return self.reg["L1"] * torch.abs(self.fc.weight).sum()

    def L2(self):
        return self.reg["L2"] * torch.sum(self.fc.weight ** 2)

    def forward(self, x):
        x = self.fc(x).reshape(-1, self.output_size, self.output_size)
        return x


class LinearDropout_ImgReconstructor(Base_ImgReconstructor):
    """
    Linear model with dropout
    """

    def __init__(self, config):
        super().__init__(config)
        self.fc_block = bl.FC_block(
            self.input_size,
            self.output_size ** 2,
            dropout_rate=config["dropout_rate"],
            activation=config["act_f"],
            batchnorm=config["batchnorm"],
        )

    def forward(self, x):
        return self.fc_block(x).view(-1, self.output_size, self.output_size)


class LinearDropoutBefore_ImgReconstructor(Base_ImgReconstructor):
    """
    Linear model with dropout
    """

    def __init__(self, config):
        super().__init__(config)
        self.block = nn.Sequential(
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(self.input_size, self.output_size ** 2),
            bl.act_func([config["act_f"]]),
        )

    def forward(self, x):
        return self.block(x).view(-1, self.output_size, self.output_size)


class Zhang_ImgReconstructor(Base_ImgReconstructor):
    """
    FC+AE model adapted from https://arxiv.org/abs/1904.13007

    config file arguments allow for some 
    flexibility that diverges from the original model
    
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        keys = list(config.keys())
        h_sizes = sorted(list(filter(re.compile("h.*size").match, keys)))
        if h_sizes == None:
            h_sizes = []
        conv_ch = sorted(list(filter(re.compile("conv.*_ch").match, keys)))
        conv_ks = sorted(list(filter(re.compile("conv.*_ks").match, keys)))
        FC_sizes = (
            [config["input_size"]]
            + [config[size] for size in h_sizes]
            + [config["intermediate_img_channels"] * config["output_size"] ** 2]
        )
        ENC_ch = [config["intermediate_img_channels"]] + [config[ch] for ch in conv_ch]
        ENC_ks = [config[ks] for ks in conv_ks]
        DEC_ch = ENC_ch[::-1][:-1] + [1]
        DEC_ks = ENC_ks[::-1]
        self.FC2img = nn.Sequential(
            *[
                bl.FC_block(
                    in_f,
                    out_f,
                    activation=config["act_f"],
                    batchnorm=config["batchnorm"],
                    dropout_rate=config["dropout_FC"],
                    order=config["order_FC"],
                )
                for in_f, out_f in zip(FC_sizes[:-1], FC_sizes[1:])
            ]
        )
        self.ConvEncoder = nn.Sequential(
            *[
                bl.Conv2d_block(
                    in_c,
                    out_c,
                    ks,
                    activation=config["act_f"],
                    batchnorm=config["batchnorm"],
                    dropout_rate=config["dropout_CE"],
                    order=config["order_CE"],
                    stride=2,
                    padding=int(ks // 2),
                )
                for in_c, out_c, ks in zip(ENC_ch[:-1], ENC_ch[1:], ENC_ks)
            ]
        )

        ConvDecoderLayers = utils.intersperse(
            [
                bl.Conv2d_block(
                    in_c,
                    out_c,
                    ks,
                    activation=config["act_f"],
                    batchnorm=config["batchnorm"],
                    dropout_rate=config["dropout_CD"],
                    order=config["order_CD"],
                    padding=int(ks // 2),
                )
                for in_c, out_c, ks in zip(DEC_ch[:-1], DEC_ch[1:], DEC_ks)
            ],
            nn.Upsample(scale_factor=2),
            item_end=False,
            item_first=True,
        )
        if "last_layer_act_f" in config.keys():
            ConvDecoderLayers[-1] = bl.Conv2d_block(
                in_c=DEC_ch[-2],
                out_c=DEC_ch[-1],
                kernel_size=DEC_ks[-1],
                activation=config["last_layer_act_f"],
                padding=int(DEC_ks[-1] // 2),
            )
        self.ConvDecoder = nn.Sequential(*ConvDecoderLayers)

    def forward(self, x):
        x = self.FC2img(x).view(
            -1,
            self.config["intermediate_img_channels"],
            self.config["output_size"],
            self.config["output_size"],
        )
        x = self.ConvEncoder(x)
        x = self.ConvDecoder(x)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return x


class FC_CNN_ImgReconstructor(Base_ImgReconstructor):
    """
    FC+CNN model
    """

    def __init__(self, config):
        """ Init model according to configuration dictionary

        Args:
            config (dict): dictionary defining parameters of the model
            
            In particular the following keys correspond to:

            input_size -> number of neurons/responses to decode
            FC_h.*size -> size of hidden layers in the FC block
            intermediate_img_channels -> number of channels of the intermediate img / fist 2d feature map / input to CNN block
            CNN_ch, CNN_ks  -> channels and kernel sizes of CNN layers 
            *_act_f, *_dropout, *_batchnorm -> activation function, dropout rate and batchnorm to apply to each layer of * block
            *_order_in_block -> define in which order to apply activation function (a), dropout (d) and batchnorm (b) in each layer of the * block. Example: 'abd'
            LL_* -> last layer * property            

        """
        super().__init__(config)
        self.config = config
        keys = list(config.keys())
        h_sizes = sorted(list(filter(re.compile("FC_h.*size").match, keys)))
        if h_sizes == None:
            h_sizes = []
        FC_sizes = (
            [config["input_size"]]
            + [config[size] for size in h_sizes]
            + [config["intermediate_img_channels"] * config["output_size"] ** 2]
        )
        CNN_ch_list = (
            [config["intermediate_img_channels"]]
            + [config["CNN_ch"]] * (config["CNN_layers"] - 1)
            + [1]
        )
        CNN_ks_list = [config["CNN_ks"]] * config["CNN_layers"]
        CNN_batchnorm_list = [config["CNN_batchnorm"]] * (config["CNN_layers"] - 1) + [
            config["LL_batchnorm"]
        ]
        CNN_dropout_list = [config["CNN_dropout"]] * (config["CNN_layers"] - 1) + [
            False
        ]
        CNN_act_f_list = [config["CNN_act_f"]] * (config["CNN_layers"] - 1) + [
            config["LL_act_f"]
        ]

        self.FC2img = nn.Sequential(
            *[
                bl.FC_block(
                    in_f,
                    out_f,
                    activation=config["FC_act_f"],
                    batchnorm=config["FC_batchnorm"],
                    dropout_rate=config["FC_dropout"],
                    order=config["FC_order_in_block"],
                )
                for in_f, out_f in zip(FC_sizes[:-1], FC_sizes[1:])
            ]
        )

        CNN_layers = [
            bl.Conv2d_block(
                in_c,
                out_c,
                ks,
                activation=act_f,
                batchnorm=bn,
                dropout_rate=do_rate,
                order=config["CNN_order_in_block"],
                stride=1,
                padding=int(ks // 2),
            )
            for in_c, out_c, ks, act_f, bn, do_rate in zip(
                CNN_ch_list[:-1],
                CNN_ch_list[1:],
                CNN_ks_list,
                CNN_act_f_list,
                CNN_batchnorm_list,
                CNN_dropout_list,
            )
        ]
        self.CNN_block = nn.Sequential(*CNN_layers)

    def forward(self, x):
        x = self.FC2img(x).view(
            -1,
            self.config["intermediate_img_channels"],
            self.config["output_size"],
            self.config["output_size"],
        )
        x = self.CNN_block(x)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return x


class Conv_AE(nn.Module):
    def __init__(
        self,
        ch_list,
        ks_list,
        act_f="relu",
        batchnorm=True,
        dropout_rate=False,
        order="adb",
    ):
        super().__init__()
        self.Conv1 = bl.Conv2d_block(
            ch_list[0],
            ch_list[1],
            ks_list[0],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[0] // 2),
        )
        self.Conv2 = bl.Conv2d_block(
            ch_list[1],
            ch_list[2],
            ks_list[1],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[1] // 2),
        )
        self.Conv3 = bl.Conv2d_block(
            ch_list[2],
            ch_list[3],
            ks_list[2],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[2] // 2),
        )
        self.pool = nn.MaxPool2d(2)
        self.TConv1 = bl.ConvTranspose2d_block(
            ch_list[3],
            ch_list[2],
            ks_list[2],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[2] // 2),
            stride=2,
            output_padding=1,
        )
        self.TConv2 = bl.ConvTranspose2d_block(
            ch_list[2],
            ch_list[1],
            ks_list[1],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[1] // 2),
            stride=2,
            output_padding=1,
        )
        self.TConv3 = bl.ConvTranspose2d_block(
            ch_list[1],
            1,
            ks_list[0],
            activation="sigm",
            batchnorm=False,
            dropout_rate=0,
            order=order,
            padding=int(ks_list[0] // 2),
            stride=2,
            output_padding=1,
        )

    def forward(self, x):
        x = self.pool(self.Conv1(x))
        x = self.pool(self.Conv2(x))
        x = self.pool(self.Conv3(x))
        x = self.TConv1(x)
        x = self.TConv2(x)
        x = self.TConv3(x)
        return x


class SkipConnections_Conv_AE(nn.Module):
    def __init__(
        self,
        ch_list,
        ks_list,
        act_f="relu",
        batchnorm=True,
        dropout_rate=False,
        order="adb",
    ):
        super().__init__()
        self.Conv1 = bl.Conv2d_block(
            ch_list[0],
            ch_list[1],
            ks_list[0],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[0] // 2),
        )
        self.Conv2 = bl.Conv2d_block(
            ch_list[1],
            ch_list[2],
            ks_list[1],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[1] // 2),
        )
        self.Conv3 = bl.Conv2d_block(
            ch_list[2],
            ch_list[3],
            ks_list[2],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[2] // 2),
        )
        self.pool = nn.MaxPool2d(2)
        self.TConv1 = bl.ConvTranspose2d_block(
            ch_list[3],
            ch_list[2],
            ks_list[2],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[2] // 2),
            stride=2,
            output_padding=1,
        )
        self.TConv2 = bl.ConvTranspose2d_block(
            ch_list[2] * 2,
            ch_list[1],
            ks_list[1],
            activation=act_f,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            order=order,
            padding=int(ks_list[1] // 2),
            stride=2,
            output_padding=1,
        )
        self.TConv3 = bl.ConvTranspose2d_block(
            ch_list[1] * 2,
            1,
            ks_list[0],
            activation="sigm",
            batchnorm=False,
            dropout_rate=0,
            order=order,
            padding=int(ks_list[0] // 2),
            stride=2,
            output_padding=1,
        )

    def forward(self, x0):
        x1 = self.pool(self.Conv1(x0))  # 64
        x2 = self.pool(self.Conv2(x1))  # 128
        y = self.pool(self.Conv3(x2))  # 256
        z2 = self.TConv1(y)  # 128
        z1 = self.TConv2(torch.cat([z2, x2], 1))  # 128 + 128
        z0 = self.TConv3(torch.cat([z1, x1], 1))
        return z0


class FC_AE(Base_ImgReconstructor):
    def __init__(self, config):
        super().__init__(config)
        keys = list(config.keys())
        h_sizes = sorted(list(filter(re.compile("FC_h.*size").match, keys)))
        if h_sizes == None:
            h_sizes = []
        FC_sizes = (
            [config["input_size"]]
            + [config[size] for size in h_sizes]
            + [config["intermediate_img_channels"] * config["output_size"] ** 2]
        )
        self.FC2img = nn.Sequential(
            *[
                bl.FC_block(
                    in_f,
                    out_f,
                    activation=config["FC_act_f"],
                    batchnorm=config["FC_batchnorm"],
                    dropout_rate=config["FC_dropout"],
                    order=config["FC_order"],
                )
                for in_f, out_f in zip(FC_sizes[:-1], FC_sizes[1:])
            ]
        )

        conv_ch = sorted(list(filter(re.compile("conv.*_ch").match, keys)))
        conv_ks = sorted(list(filter(re.compile("conv.*_ks").match, keys)))
        ch_list = [config["intermediate_img_channels"]] + [config[ch] for ch in conv_ch]
        ks_list = [config[ks] for ks in conv_ks]
        self.AE = self.AE = Conv_AE(
            ch_list,
            ks_list,
            config["AE_act_f"],
            batchnorm=config["AE_batchnorm"],
            dropout_rate=config["AE_dropout"],
            order=config["AE_order"],
        )

    def forward(self, x):
        x = self.FC2img(x).view(
            -1,
            self.config["intermediate_img_channels"],
            self.config["output_size"],
            self.config["output_size"],
        )
        x = self.AE(x)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        return x


class FC_SkipAE(FC_AE):
    def __init__(self, config):
        super().__init__(config)
        keys = list(config.keys())
        conv_ch = sorted(list(filter(re.compile("conv.*_ch").match, keys)))
        conv_ks = sorted(list(filter(re.compile("conv.*_ks").match, keys)))
        ch_list = [config["intermediate_img_channels"]] + [config[ch] for ch in conv_ch]
        ks_list = [config[ks] for ks in conv_ks]
        self.AE = SkipConnections_Conv_AE(
            ch_list,
            ks_list,
            config["AE_act_f"],
            batchnorm=config["AE_batchnorm"],
            dropout_rate=config["AE_dropout"],
            order=config["AE_order"],
        )

    def forward(self, x):
        return super().forward(x)


#%%
