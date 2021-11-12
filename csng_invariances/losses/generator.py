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
