"""Module providing neuron selection functionality."""

import torch


def score(dnn_single_neuron_correlations, linear_filter_single_neuron_correlaitons):
    """Compute the neuron selection score

    Args:
        dnn_single_neuron_correlations (Tensor): tensor of single neuron
            correlations when using the encoding model
        linear_filter_single_neuron_correlaitons (Tensor]): tensor of single
            neuron correlations when using the linear filters

    Returns:
        Tensor: Score tensor
    """
    return (
        1 - linear_filter_single_neuron_correlaitons
    ) * dnn_single_neuron_correlations


def select_neurons(score, num_neurons):
    """Select num_neurons higest scoring neurons.

    Args:
        score (Tensor): Score tensor
        num_neurons (int): number of neurons to select.

    Returns:
        Tensor: Tensor of select
    """
    _, indicies = torch.sort(score, descending=True, stable=True)
    return indicies[0:num_neurons]
