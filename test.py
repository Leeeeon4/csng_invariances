# %%
import torch
import torch.optim as optim
import wandb
import datetime

from pathlib import Path
from rich import print
from torch import nn

from csng_invariances.datasets.lurz2020 import download_lurz2020_data, static_loaders
from csng_invariances.models.discriminator import (
    download_pretrained_lurz_model,
    se2d_fullgaussian2d,
)
from csng_invariances.training.trainers import standard_trainer as lurz_trainer
from csng_invariances.utility.data_helpers import save_configs, load_configs

#%%
d = {"dataset": "Lurz"}
d["dataset"]

# %%
# to be done by argparsing
batch_size = 64
seed = 1
interval = 1
patience = 5
lr_init = 0.005
max_iter = 200
tolerance = 1e-6
lr_decay_steps = 3
lr_decay_factor = 0.3
min_lr = 0.0001
detach_core = True
# %%
# read from system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = False if str(device) == "cpu" else True

# %%
# Settings
lurz_data_directory = Path.cwd() / "data" / "external" / "lurz2020"
lurz_model_directory = Path.cwd() / "models" / "external" / "lurz2020"
lurz_model_path = lurz_model_directory / "transfer_model.pth.tar"
dataset_config = {
    "paths": [str(lurz_data_directory / "static20457-5-9-preproc0")],
    "batch_size": batch_size,
    "seed": seed,
    "cuda": cuda,
    "normalize": True,
    "exclude": "images",
}
model_config = {
    "init_mu_range": 0.55,
    "init_sigma": 0.4,
    "input_kern": 15,
    "hidden_kern": 13,
    "gamma_input": 1.0,
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 0,
        "final_tanh": False,
    },
    "gamma_readout": 2.439,
}
trainer_config = {
    "avg_loss": False,
    "scale_loss": True,
    "loss_function": "PoissonLoss",
    "stop_function": "get_correlations",
    "loss_accum_batch_n": None,
    "verbose": True,
    "maximize": True,
    "restore_best": True,
    "cb": None,
    "track_training": True,
    "return_test_score": False,
    "epoch": 0,
    "device": device,
    "seed": seed,
    "detach_core": detach_core,
    "batch_size": batch_size,
    "lr_init": lr_init,
    "lr_decay_factor": lr_decay_factor,
    "lr_decay_steps": lr_decay_steps,
    "min_lr": min_lr,
    "max_iter": max_iter,
    "tolerance": tolerance,
    "interval": interval,
    "patience": patience,
}
configs = {
    "dataset_config": dataset_config,
    "model_config": model_config,
    "trainer_config": trainer_config,
}
# %%
t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
readout_model_directory = Path.cwd() / "models" / "encoding" / t
readout_model_directory.mkdir(parents=True, exist_ok=True)

save_configs(configs, readout_model_directory)
# directory = "/home/leon/csng_invariances/models/encoding/2021-11-04_10:15:53"
configs = load_configs(readout_model_directory)
print(configs)
# %%
# Load data and model
download_lurz2020_data() if (
    lurz_data_directory / "README.md"
).is_file() is False else None
download_pretrained_lurz_model() if (lurz_model_path).is_file() is False else None

# %%
print(f"Running current dataset config:\n{dataset_config}")
dataloaders = static_loaders(**dataset_config)
# %%
print(f"Running current model config:\n{model_config}")
# build model
model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
# load state_dict of pretrained core
transfer_model = torch.load(lurz_model_path, map_location=device)
model.load_state_dict(transfer_model, strict=False)
# %%
transfer_model
# %%
# Training readout
print(f"Running current training config:\n{trainer_config}")
wandb.init()
config = wandb.config
kwargs = dict(dataset_config, **model_config)
kwargs.update(trainer_config)
config.update(kwargs)
score, output, model_state = lurz_trainer(
    model=model, dataloaders=dataloaders, **trainer_config
)
# %%
# Saving model (core + readout)
torch.save(
    model.state_dict(), readout_model_directory / "Pretrained_core_readout_lurz.pth"
)
# to load:
# model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
# model.load_state_dict(torch.load(read_model_path / "Pretrained_core_readout_lurz.pth"))
# %%
# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#%%
# Generator Code


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, (4, 4), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (2, 4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (2, 4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, (2, 4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, (4, 4), 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# %%
tensor = torch.randn(64, nz, 1, 1, dtype=torch.float, device=device)
# output dim if flattened
a = nn.ConvTranspose2d(nz, 1, (36, 64)).to(device)
# %%
n = a(tensor)
n.shape
# %%
mod = Generator(1).to(device)
fw = mod.forward(tensor)
fw.shape
# %%
fw = torch.randn(64, 1, 36, 64, dtype=torch.float, device=device)
# %%
res = model.forward(fw)
res.shape
# %%
class ExampleGAN:
    """Create gan model class.

    Create gan model and hold it for usage in the experiments.
    This class is based on Lazarou 2020: PyTorch and GANs: A Micro Tutorial.
    tutorial, which can be found at:
    https://towardsdatascience.com/\
        pytorch-and-gans-a-micro-tutorial-804855817a6b
    Most in-line comments are direct quotes from the tutorial. The code is
    copied and slightly adapted.
    """

    def __init__(
        self,
        generator,
        discriminator,
        noise_fn,
        data_fn,
        batch_size=32,
        device="cpu",
        lr_d=1e-3,
        lr_g=2e-4,
    ):
        """A GAN class for holding and training a generator and discriminator.

        Args:
            generator (torch.nn.Module): A Generator network.
            discriminator (torch.nn.Module): A Discriminator network.
            noise_fn (function f(num: int)): A noise function.
                This is the function used to sample latent vectors Z,
                which our Generator will map to generated samples X.
                This function must accept an integer num as input and
                return a 2D Torch tensor with shape (num, latent_dim).
            data_fn (function f(num: int)): A data function.
                This is the function that our Generator is tasked with
                learning. This function must accept an integer num as
                input and return a 2D Torch tensor with shape (num, data_dim),
                where data_dim is the dimension of the data we are trying
                to generate, the input_dim of our Discriminator.
            batch_size (int, optional): training batch size. Defaults to 32.
            device (string, optional): cpu or CUDA. Defaults to cpu.
            lr_d (float, optional): learning rate for the discriminator.
                Defaults to 1e-3.
            lr_g (float, optional): learning rate for the generator.
                Defaults to 2e-4.
        """
        self.generator = generator
        self.generator = self.generator.to(device)
        self.discriminator = discriminator
        self.discriminator = self.discriminator.to(device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.device = device
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.betas = (0.5, 0.999)
        #       The optimization criterion used is binary cross entropy loss.
        self.criterion = nn.BCELoss()
        #       An optimizer for the discriminator part of the model.
        self.optim_d = optim.Adam(
            discriminator.parameters(), lr=self.lr_d, betas=self.betas
        )
        #       An optimizer for the generator part of the model.
        self.optim_g = optim.Adam(
            generator.parameters(), lr=self.lr_g, betas=self.betas
        )
        #       A vector of ones, the discriminator target.
        self.target_ones = torch.ones((batch_size, 1)).to(device)
        #       A vector of zeros, the generators target.
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)

    def generate_samples(self, latent_vec=None, num=None):
        """Generate samples from the generator.

        If latent_vec and num are None then us self.batch_size random latent
        vectors.

        Args:
            latent_vec (torch.Tensor, optional): A pytorch latent vector or
                None. Defaults to None.
            num (int, optional): The number of samples to generate if
                latent_vec is None. Defaults to None.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        #       no_grad tells torch not to compute gradients here
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        #       Clear the gradients. The coolest thing about PyTorch is that the
        #       gradient automatically accumulates in each parameter as the network is
        #       used. However, we typically want to clear these gradients between each
        #       step of the optimizer; the zero_grad method does just that.
        self.generator.zero_grad()

        #       Sample batch_size latent vectors from the noise-generating function.
        latent_vec = self.noise_fn(self.batch_size)

        #       Feed the latent vectors into the Generator and get the generated
        #       samples as output (under the hood, the generator.forward method is
        #       called here). Remember, PyTorch is define-by-run, so this is the point
        #       where the generator’s computational graph is built.
        generated = self.generator(latent_vec)

        #       Feed the generated samples into the Discriminator and get its
        #       confidence that each sample is real. Remember, the Discriminator is
        #       trying to classify these samples as fake (0) while the Generator is
        #       trying trick it into thinking they’re real (1). Just as in the previous
        #       line, this is where the Discriminator’s computational graph is built,
        #       and because it was given the generated samples generated as input, this
        #       computational graph is stuck on the end of the Generator’s
        #       computational graph.
        classifications = self.discriminator(generated)

        #       Calculate the loss for the Generator. Our loss function is
        #       Binary Cross Entropy, so the loss for each of the batch_size samples is
        #       calculated and averaged into a single value. loss is a PyTorch tensor
        #       with a single value in it, so it’s still connected to the full
        #       computational graph.
        loss = self.criterion(classifications, self.target_ones)

        #       This is where the magic happens. Or rather, this is where the prestige
        #       happens, since the magic has been happening invisibly this whole time.
        #       Here, the backward method calculates the gradient d_loss/d_x for every
        #       parameter x in the computational graph.
        loss.backward()

        #       Apply one step of the optimizer, nudging each parameter down the
        #       gradient. If you’ve built a GAN in Keras before, you’re probably
        #       familiar with having to set my_network.trainable = False. One of the
        #       advantages of PyTorch is that you don’t have to bother with that,
        #       because optim_g was told to only concern itself with our Generator’s
        #       parameters.
        self.optim_g.step()

        #       Return the loss. We will be storing these in a list for later
        #       visualization. However, it’s vital that we use the item method to
        #       return it as a float, not as a PyTorch tensor. This is because, if we
        #       keep a reference to that tensor object in a list, Python will also hang
        #       on to the entire computational graph. This is a big waste of memory,
        #       so we need to make sure that we only keep what we need (the value)
        #       so that Python’s garbage collector can clean up the rest.
        return loss.item()

    def train_step_discriminator(self):
        """Train the discriminator one step and return the losses."""
        #       Clear the gradients. The coolest thing about PyTorch is that the
        #       gradient automatically accumulates in each parameter as the network is
        #       used. However, we typically want to clear these gradients between each
        #       step of the optimizer; the zero_grad method does just that.
        self.discriminator.zero_grad()

        #       real samples
        #       Sample some real samples from the target function, get the
        #       Discriminator’s confidences that they’re real (the Discriminator wants
        #       to maximize this!), and calculate the loss. This is very similar to the
        #       generator’s training step.
        real_samples = self.data_fn(self.batch_size)
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        #       generated samples
        #       Sample some generated samples from the generator, get the
        #       Discriminator’s confidences that they’re real (the Discriminator wants
        #       to minimize this!), and calculate the loss. Because we’re training the
        #       Discriminator here, we don’t care about the gradients in the Generator
        #       and as such we use the no_grad context manager. Alternatively, you
        #       could ditch the no_grad and substitute in the line
        #       pred_fake = self.discriminator(fake_samples.detach())
        #       and detach fake_samples from the Generator’s computational graph after
        #       the fact, but why bother calculating it in the first place?
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        #       combine
        #       Average the computational graphs for the real samples and the generated
        #       samples. Yes, that’s really it. This is my favourite line in the whole
        #       script, because PyTorch is able to combine both phases of the
        #       computational graph using simple Python arithmetic.
        loss = (loss_real + loss_fake) / 2

        #       Calculate the gradients, apply one step of gradient descent, and return
        #       the losses.
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_step(self):
        """Train both networks and return the losses."""
        loss_d = self.train_step_discriminator()
        loss_g = self.train_step_generator()
        return loss_g, loss_d


# %%
generator = ExampleGenerator(126)
generator.forward()
