"""Module for generator model classes."""

import torch
from typing import OrderedDict, Tuple
from math import log


class Generator(torch.nn.Module):
    """Highlevel GeneratorModel class."""

    def __init__(
        self,
        output_shape: torch.Size = (64, 1, 36, 64),
        latent_space_dimension: int = 128,
        batch_size: int = None,
        device=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            output_shape ()
            latent_space_dimension (int): Size of the latent vector
            batch_size (int): batch_size
            device (str, optional): Device to compute on. Defaults to None.
        """
        super().__init__()
        self.output_shape = output_shape
        if batch_size is not None:
            self.output_shape[0] = self.batch_size
        self.batch_size, self.channels, self.height, self.width = self.output_shape
        self.latent_space_dimension = latent_space_dimension
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class GrowingLinearGenerator(Generator):
    def __init__(
        self,
        output_shape: Tuple[int, int, int, int] = (64, 1, 36, 64),
        latent_space_dimension: int = 128,
        batch_size: int = None,
        device: str = None,
        layer_growth: float = 1.1,
        *args,
        **kwargs,
    ):
        super().__init__(
            output_shape=output_shape,
            latent_space_dimension=latent_space_dimension,
            batch_size=batch_size,
            device=device,
            *args,
            **kwargs,
        )
        self.layer_growth = layer_growth

        quotient, remainder = divmod(
            log(
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
            l = torch.nn.Linear(
                in_features=int(
                    self.latent_space_dimension * self.layer_growth ** layer
                ),
                out_features=int(
                    self.latent_space_dimension * self.layer_growth ** (layer + 1)
                ),
                device=device,
            )
            self.layers[f"Linear Layer {layer}"] = l
            self.layers[f"ReLU {layer}"] = torch.nn.ReLU()
        if remainder != 0:
            l = torch.nn.Linear(
                in_features=int(
                    self.latent_space_dimension * self.layer_growth ** (quotient)
                ),
                out_features=int(self.channels * self.height * self.width),
                device=device,
            )
            self.layers["Last Linear Layer"] = l
            self.layers["Last ReLU"] = torch.nn.ReLU()

        self.linear_stack = torch.nn.Sequential(self.layers)

    def forward(self, x):
        x = self.linear_stack(x)
        return super().forward(x)
