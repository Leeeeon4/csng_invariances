"""Module providing generator loss functions."""

import torch
import torch.nn as nn


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


class SelectedNeuronActivityWithDiffernceInImage(nn.Module):
    """Loss function which maximizes neural activation and difference_scaling * difference in images."""

    def __init__(self, difference_scaling) -> None:
        super().__init__()
        self.difference_scaling = difference_scaling

    def forward(
        self, activations: torch.Tensor, images: torch.Tensor, neuron_idx: int
    ) -> torch.Tensor:
        """Return the scalar neuron activation of the selected neuron.

        Args:
            activations (torch.Tensor): input activation tensor.
            images (torch.Tensor): input image tensor. Image, which is passed to encoding model.
            neuron_idx (int): neuron to select

        Returns:
            torch.Tensor: output
        """
        return -(
            activations[:, neuron_idx].sum()
            + self.difference_scaling * (images.diff(dim=0))
        )


if __name__ == "__main__":
    pass
