# %%
import torch
import wandb
import datetime

from pathlib import Path
from rich import print

from csng_invariances.datasets.lurz2020 import download_lurz2020_data, static_loaders
from csng_invariances.models.discriminator import (
    download_pretrained_lurz_model,
    se2d_fullgaussian2d,
)
from csng_invariances.training.trainers import standard_trainer as lurz_trainer


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
lurz_data_path = Path.cwd() / "data" / "external" / "lurz2020"
lurz_model_path = Path.cwd() / "models" / "external" / "lurz2020"
dataset_config = {
    "paths": [str(lurz_data_path / "static20457-5-9-preproc0")],
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
# %%
# Load data and model
download_lurz2020_data() if (lurz_data_path / "README.md").is_file() is False else None
download_pretrained_lurz_model() if (
    lurz_model_path / "transfer_model.pth.tar"
).is_file() is False else None

# %%
print(f"Running current dataset config:\n{dataset_config}")
dataloaders = static_loaders(**dataset_config)
# %%
print(f"Running current model config:\n{model_config}")
# build model
model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
# load state_dict of pretrained core
transfer_model = torch.load(
    Path.cwd() / "models" / "external" / "lurz2020" / "transfer_model.pth.tar",
    map_location=device,
)
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
t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
readout_model_path = Path.cwd() / "models" / "encoding" / t
readout_model_path.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), readout_model_path / "Pretrained_core_readout_lurz.pth")
# to load:
# model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
# model.load_state_dict(torch.load(read_model_path / "Pretrained_core_readout_lurz.pth"))
# %%
for imgs, resps in dataloaders["train"]["20457-5-9-0"]:
    None

# %%
for i in range(imgs.shape[0]):
    img = imgs[i, :, :, :]
    resp = resps[i, :]
    img = img.reshape(1, 1, 36, 64)
    prediction = model.forward(img)
    print(f"Shape of image: {img.shape}")
    print(f"Shape of prediction: {prediction.shape}")
    print(f"Shape of response: {resp.shape}")
    print(f"Summed image: {torch.sum(img)}")
    print(f"Summed prediction: {torch.sum(prediction)}")
    print(f"Summed response: {torch.sum(resp)}")
    print(f"Summed difference: {torch.sum(prediction-resp)}")
# %%
dim = tuple(img.shape)
dim
# %%
rand_img = torch.randint(255, dim, device=device, dtype=torch.float)
# %%
model.forward(rand_img)
# %%
tensor = torch.randn(size=(1, 128), device=device)
a = torch.nn.ConvTranspose2d(in_channels=1, out_channels=36, kernel_size=(1, 1))
a(tensor)
# %%
from torch import nn


class ExampleGenerator(nn.Module):
    """Create generator model class.

    Create generator model to generate sample form latent space.
    Model consists of: linear layer, leaky ReLU, linear layer, linear layer,
    and optionally passed output_activation.
    This class is based on Lazarou 2020: PyTorch and GANs: A Micro Tutorial.
    tutorial, which can be found at:
    https://towardsdatascience.com/\
        pytorch-and-gans-a-micro-tutorial-804855817a6b
    Most in-line comments are direct quotes from the tutorial. The code is
    copied and slightly adapted.
    """

    def __init__(self, latent_dim, output_activation=None):
        """Instantiates the generator model.

        Args:
            latent_dim (int): Dimension of the latent space tensor.
            output_activation (torch.nn activation funciton, optional):
                activation function to use. Defaults to None.
        """
        super().__init__()
        # TODO Output dimension
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(self.latent_dim, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

        self.output_activation = output_activation

    def forward(self, input_tensor):
        """Forward pass.

        Defines the forward pass through the generator model.
        Maps the latent vector to samples.

        Args:
            input_tensor (torch.Tensor): tensor which is input
                into the generator model.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        # TODO Output dimension
        intermediate = self.linear1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear3(intermediate)
        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
        return intermediate

    def generate_batch(
        self, distribution="normal", mean=0, sdv=1, width=2, batch_size=64
    ):
        """Generate a batch of samples.

        Args:
            distribution (str, optional): Type of distribution to sample from,
                option are: "normal", "uniform". Defaults to "normal".
            mean (float, optional): Mean of distribution.
                Defaults to 0.
            sdv (float, optional): Standard deviation of normal distribution.
                Defaults to 1.
            width (float, optional): Total width of uniform distribution.
                I.e. a width of 2 means 1 above mean and 1 below. Defaults to 2.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            Tensor: Sampletensor of dimension (batch_size, latent_dim)
        """
        assert (
            distribution is "normal" or "uniform"
        ), "Distribution type unknown. Did you mean 'normal' or 'uniform'?"
        self.distribution = distribution
        self.mean = mean
        self.sdv = sdv
        self.width = width
        self.batch_size = batch_size
        if self.distribution is "normal":
            self.batch_tensor = torch.normal(
                mean=self.mean, std=self.sdv, size=(self.batch_size, self.latent_dim)
            )
        elif self.distribution is "uniform":
            batch_tensor = torch.rand(size=(self.batch_size, self.latent_dim))
            self.batch_tensor = batch_tensor.add((self.mean - 1) / 2) * self.width
        return self.batch_tensor


# %%
import torch.optim as optim


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
