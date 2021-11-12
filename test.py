"""Testing scipt"""
#%%
from typing import OrderedDict
from numpy.core.fromnumeric import size
import torch
from torch import nn
import math
from torch._C import Module

from torch.nn.modules import activation

from csng_invariances.data.preprocessing import image_preprocessing
from csng_invariances.encoding import *

# %%
from csng_invariances.data._data_helpers import *

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
encoding_model = load_encoding_model(
    "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
)

#%%
class Generator(nn.Module):
    """Highlevel GeneratorModel class."""

    def __init__(
        self,
        images,
        responses,
        latent_space_dimension,
        batch_size,
        device=None,
        *args,
        **kwargs,
    ):
        """Instantiation.

        Args:
            images (Tensor): image tensor.
            responses (Tensor): response tensor.
            latent_space_dimension (int): Size of the latent vector
            batch_size (int): batch_size
            device (str, optional): Device to compute on. Defaults to None.
        """
        super().__init__()

        self.images = images
        self.responses = responses
        self.latent_space_dimension = latent_space_dimension
        self.batch_size = batch_size
        self.image_count, self.channels, self.height, self.width = self.images.shape
        _, self.neuron_count = self.responses.shape
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def forward(self, x):
        """Forward pass through the model

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: tensor representing a generated image.
        """
        # reshape if necessary
        if x.shape != (x.shape[0], self.channels, self.height, self.width):
            x = x.reshape(x.shape[0], self.channels, self.height, self.width)
        return x

    def _empty_sample_tensor(self):
        return torch.empty(
            size=(self.batch_size, self.latent_space_dimension),
            device=self.device,
            dtype=torch.float,
            requires_grad=True,
        )

    def sample_from_normal(self, mean=0, std=1):
        return nn.init.normal_(self._empty_sample_tensor(), mean, std)

    def sample_from_unit(self, low, high):
        return nn.init.uniform_(self._empty_sample_tensor(), low, high)


class LinearGenerator(Generator):
    def __init__(
        self,
        images,
        responses,
        latent_space_dimension,
        batch_size,
        layer_growth,
        device=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            images,
            responses,
            latent_space_dimension,
            batch_size,
            device=device,
            *args,
            **kwargs,
        )
        self.layer_growth = layer_growth

        quotient, remainder = divmod(
            math.log(
                (
                    (self.channels * self.height * self.width)
                    / self.latent_space_dimension
                ),
                self.layer_growth,
            ),
            1,
        )
        self.layers = OrderedDict()
        for layer in range(int(quotient)):
            l = nn.Linear(
                in_features=int(
                    self.latent_space_dimension * self.layer_growth ** layer
                ),
                out_features=int(
                    self.latent_space_dimension * self.layer_growth ** (layer + 1)
                ),
                device=device,
            )
            self.layers[f"Linear Layer {layer}"] = l
            self.layers[f"ReLU {layer}"] = nn.ReLU()
        if remainder != 0:
            l = nn.Linear(
                in_features=int(
                    self.latent_space_dimension * self.layer_growth ** (quotient)
                ),
                out_features=int(self.channels * self.height * self.width),
                device=device,
            )
            self.layers["Last Linear Layer"] = l
            self.layers["Last ReLU"] = nn.ReLU()

        self.linear_stack = nn.Sequential(self.layers)

    def forward(self, x):
        x = self.linear_stack(x)
        return super().forward(x)

    def sample_from_normal(self, mean=0, std=1):
        return super().sample_from_normal(mean, std)

    def sample_from_unit(self, low, high):
        return super().sample_from_unit(low, high)


class ComputeModel(nn.Module):
    def __init__(self, generator_model, encoding_model, *args, **kwargs):
        super().__init__()
        self.generator_model = generator_model
        self.encoding_model = encoding_model.requires_grad_(False)

    def forward(self, x):
        x = generator_model(x)
        x = image_preprocessing(x)
        x = encoding_model(x)
        return x


class SelectedNeuronActivation(nn.Module):
    """Naive loss function which maximizes the selected neuron's activation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, neuron_idx: int) -> torch.Tensor:
        """Return the scalar neuron activation of the selected neuron.

        Args:
            inputs (torch.Tensor): input activation tensor.
            neuron_idx (int): neuron to select

        Returns:
            torch.Tensor: output
        """
        return -inputs[:, neuron_idx].sum()


#%%
batch_size = 64
images = torch.randn(size=(batch_size, 1, 36, 64), device=device, dtype=torch.float)
responses = torch.randn(size=(batch_size, 5335), device=device, dtype=torch.float)
config = {
    "images": images,
    "responses": responses,
    "latent_space_dimension": 128,
    "batch_size": batch_size,
    "encoding_model": encoding_model,
    "layer_growth": 2,
    "device": device,
}
#%%
generator_model = LinearGenerator(**config)
print(generator_model)
# %%
encoding_model.requires_grad_(False)
print(encoding_model)
#%%
latent_tensor = generator_model.sample_from_normal()
print(latent_tensor)

# %%
gan = ComputeModel(generator_model, encoding_model)

optimizer = optim.Adam(generator_model.parameters())
loss_function = SelectedNeuronActivation()

running_loss = 0.0
for epoch in range(2000):
    optimizer.zero_grad()
    activations = gan(latent_tensor)
    loss = loss_function(activations, 5)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if epoch % 200 == 0:
        print("[%d] loss: %.3f" % (epoch + 1, running_loss))
    running_loss = 0.0

# %%
image_5 = generator_model(generator_model.sample_from_normal())
image_5.shape
# %%
from matplotlib import pyplot as plt

for i in range(64):
    plt.imshow(image_5[i, :, :, :].cpu().detach().squeeze())
    plt.show()

# %%
