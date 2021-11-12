import sys

sys.path.append("/home/baroni/reconstruct_images/")
import wandb
from datamodules import *
from models import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

# Set up your default hyperparameters
default = {
    'loss' : 'mse',
    "input_size": 5000,
    "output_size": 64,
    "resize_stim": False,
    "h_1_size": 1000,
    "intermediate_img_channels": 64,
    "conv_1_ch": 256,
    "conv_2_ch": 256,
    "conv_3_ch": 256,
    # "conv_4_ch": 256,
    "conv_1_ks": 7,
    "conv_2_ks": 5,
    "conv_3_ks": 3,
    # "conv_4_ks": 3,
    "dropout_FC": 0.2,
    "dropout_CE": 0,
    "dropout_CD": 0,
    "act_f": "relu",
    # "last_layer_act_f": "sigm",
    "batchnorm": True,
    "order_FC": "bda",
    "order_CE": "bda",
    "order_CD": "bda",
    "lr": 0.01,
    "batch_size": 210,
    "max_epochs": 500,
}
# Pass your defaults to wandb.init
run = wandb.init(config=default, entity="csng-cuni", project="Zhang_img_rec")
# Access all hyperparameter values through wandb.config
config = wandb.config

# Set up model
model = Zhang_ImgReconstructor(config)
print(model)
file = "/home/baroni/reconstruct_images/data/stim_size=64/randomly_selecting/n_neurons=5000.pickle"
dm = StimRespDataModule(file, config)

# define logger for trainer
wandb_logger = WandbLogger()
early_stop = EarlyStopping(monitor="val/loss", patience=10)


class WandbImageCallback(pl.Callback):
    def __init__(self, dm, num_img=32):
        super().__init__()
        self.num_imgs = num_img

    def on_test_end(self, trainer, pl_module):
        imgs = next(iter(dm.test_dataloader()))[0][: self.num_imgs].to(
            device=pl_module.device
        )
        recs = pl_module(
            next(iter(dm.test_dataloader()))[1][: self.num_imgs]
            .to(device=pl_module.device)
            .float()
        )
        mosaics = torch.cat([imgs, recs], dim=-2)
        caption = "Top: stimuli, Bottom: reconstructions"
        trainer.logger.experiment.log(
            {
                "images and reconstructions": [
                    wandb.Image(mosaic, caption=caption) for mosaic in mosaics
                ],
                "global_step": trainer.global_step,
            }
        )


rec_callback = WandbImageCallback(dm)
trainer = pl.Trainer(
    callbacks=[early_stop, rec_callback],
    max_epochs=config["max_epochs"],
    gpus=[0],
    logger=wandb_logger,
    log_every_n_steps=1,
)
trainer.fit(model, dm)

# test results:
x = trainer.test(ckpt_path="best", datamodule=dm)
