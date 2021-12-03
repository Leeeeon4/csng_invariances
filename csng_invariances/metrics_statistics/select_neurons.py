"""Module providing neuron selection functionality."""

import torch
from numpy import save, load
from pathlib import Path

from csng_invariances._utils.utlis import string_time


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
    score_tensor = (
        1 - linear_filter_single_neuron_correlations
    ) * dnn_single_neuron_correlations
    t = string_time()
    directory = Path.cwd() / "reports" / "scores" / t
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "score.npy", score_tensor.cpu().numpy())
    return score_tensor


def load_score(path: str, device: str = None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data = load(path)
        data_tensor = torch.from_numpy(data)
        data_tensor.to(device)
    except Exception as err:
        print(f"An error occured. Is path correct? Is numpy file?\n\n" f"{err}")
    return data_tensor


def select_neurons(score: torch.Tensor, num_neurons: int = 50) -> list:
    """Select num_neurons higest scoring neurons.

    Args:
        score (Tensor): Score tensor
        num_neurons (int): number of neurons to select.

    Returns:
        list: list of indicies of selected neurons.
    """
    _, indicies = torch.sort(score, descending=True, stable=True)
    t = string_time()
    directory = Path.cwd() / "reports" / "neuron_selection" / t
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "selected_neuron_idxs.npy", indicies[0:num_neurons].cpu().numpy())
    return indicies[0:num_neurons].tolist()


def load_selected_neurons_idxs(path: str, device: str = None) -> list:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data = load(path).tolist()
    except Exception as err:
        print(f"An error occured. Is path correct? Is numpy file?\n\n" f"{err}")
    return data


if __name__ == "__main__":
    pass
