"""Module for generator model classes."""
#%%
from typing import OrderedDict
import torch
from torch import nn
import math


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


class GrowingLinearGenerator(Generator):
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


# TODO Load pretrained generator model

# %%
