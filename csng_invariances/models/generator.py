"""Module for generator model classes."""
#%%
import json
import torch
from torchvision.transforms import GaussianBlur
from typing import OrderedDict, Tuple
from math import log
from pathlib import Path
from csng_invariances.data._data_helpers import scale_tensor_to_0_1


class Generator(torch.nn.Module):
    """Highlevel Generator model class."""

    def __init__(
        self,
        output_shape: torch.Size = (64, 1, 36, 64),
        latent_space_dimension: int = 128,
        batch_size: int = None,
        device: str = None,
        batch_norm: bool = False,
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
        self.batch_size, self.channels, self.height, self.width = self.output_shape
        if batch_size is not None:
            self.batch_size = batch_size
        self.latent_space_dimension = latent_space_dimension
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.batch_norm = batch_norm
        self.normalization_layer = torch.nn.LazyBatchNorm2d(device=self.device)
        # torch.nn.BatchNorm2d(
        #     num_features=self.output_shape, device=self.device, dtype=torch.float
        # )

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
        # performs batch_norm if wanted
        if self.batch_norm:
            x = self.normalization_layer(x)
        x = scale_tensor_to_0_1(x) * 255
        return x

    @staticmethod
    def load_trained_generator(model_directory: str) -> Tuple[dict, torch.nn.Module]:
        """Load generator model depending on config.

        Args:
            model_directory (str): Directory in which model and config are stored.

        Returns:
            Tuple[dict, torch.nn.Module]: Tuple of config and generator_model
        """
        model_directory = Path(model_directory)
        for file in model_directory.iterdir():
            if file.suffix == ".pth":
                model_name = file.name
                print(model_name)
        with open(model_directory / "config.json", "r") as config_file:
            config = json.load(config_file)
        print(f"Config is:\n{json.dumps(config, indent=2)}")
        model_type = config["generator_model"]
        if model_type == "GrowingLinearGenerator":
            generator_model = GrowingLinearGenerator(**config)
        elif model_type == "FullyConnectedGenerator":
            generator_model = FullyConnectedGenerator(**config)
        elif model_type == "FullyConnectedGeneratorWithGaussianBlurring":
            generator_model = FullyConnectedGeneratorWithGaussianBlurring(**config)
        else:
            generator_model = Generator(**config)
        generator_model.load_state_dict(torch.load(model_directory / model_name))
        return config, generator_model


#%%
#%%
class GrowingLinearGenerator(Generator):
    """Generator architecure of linear layers which grow by factor of layer_growth
    between every layer step.
    """

    def __init__(
        self,
        layer_growth: float = 1.1,
        *args,
        **kwargs,
    ):
        super().__init__(
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


class FullyConnectedGenerator(Generator):
    """Generator Architecture as presented by Kovacs."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.layer_0_out = 512
        self.layer_1_out = 1024
        self.layer_2_out = self.channels * self.height * self.width

        self.layer_0 = torch.nn.Linear(
            in_features=self.latent_space_dimension,
            out_features=self.layer_0_out,
            device=self.device,
        )
        self.activation_function_0 = torch.nn.Tanh()
        self.layer_1 = torch.nn.Linear(
            in_features=self.layer_0_out,
            out_features=self.layer_1_out,
            device=self.device,
        )
        self.activation_function_1 = torch.nn.Tanh()
        self.layer_2 = torch.nn.Linear(
            in_features=self.layer_1_out,
            out_features=self.layer_2_out,
            device=self.device,
        )
        self.activation_function_2 = torch.nn.Tanh()

        self.linear_stack = torch.nn.Sequential(
            self.layer_0,
            self.activation_function_0,
            self.layer_1,
            self.activation_function_1,
            self.layer_2,
            self.activation_function_2,
        )

        self.linear_stack.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        First passes through linear_stack, than through forward pass for all
        generator models (super).

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.linear_stack(x)
        return super().forward(x)


class FullyConnectedGeneratorWithGaussianBlurring(FullyConnectedGenerator):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        sigma: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self.gaussian(x)


# %%
#%%
if __name__ == "__main__":
    Generator.load_trained_generator(
        "/home/leon/csng_invariances/models/generator/2021-12-08_11:34:51"
    )
# %%

# %%

# %%
