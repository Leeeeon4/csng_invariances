"""Create generator model."""

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
        self.linear1 = nn.Linear(latent_dim, 64)
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
        intermediate = self.linear1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear3(intermediate)
        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
        return intermediate
