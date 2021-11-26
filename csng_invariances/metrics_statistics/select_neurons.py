"""Module providing neuron selection functionality."""

import torch


def score(
    dnn_single_neuron_correlations: torch.Tensor,
    linear_filter_single_neuron_correlations: torch.Tensor,
) -> torch.Tensor:
    """Compute the neuron selection score

    Args:
        dnn_single_neuron_correlations (Tensor): tensor of single neuron
            correlations when using the encoding model
        linear_filter_single_neuron_correlations (Tensor): tensor of single
            neuron correlations when using the linear filters

    Returns:
        Tensor: Score tensor
    """
    return (
        1 - linear_filter_single_neuron_correlations
    ) * dnn_single_neuron_correlations


def select_neurons(score: torch.Tensor, num_neurons: int = 50) -> list:
    """Select num_neurons higest scoring neurons.

    Args:
        score (Tensor): Score tensor
        num_neurons (int): number of neurons to select.

    Returns:
        list: list of indicies of selected neurons.
    """
    _, indicies = torch.sort(score, descending=True, stable=True)
    return indicies[0:num_neurons].tolist()
