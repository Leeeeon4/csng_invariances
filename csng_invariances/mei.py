"""Module providing functions related to the generation of most exciting images."""
#%%
import json
from pathlib import Path
from numpy import load
from rich.progress import track
from rich import print
import torch
from csng_invariances.encoding import load_encoding_model
from csng_invariances.metrics_statistics.correlations import (
    compute_single_neuron_correlations_encoding_model as dnn_corrs,
)
from csng_invariances.metrics_statistics.correlations import (
    load_single_neuron_correlations_linear_filter,
)
from csng_invariances.training.mei import mei
from csng_invariances.layers.loss_function import SelectedNeuronActivation
from csng_invariances.data.datasets import (
    gaussian_white_noise_image,
    Lurz2021Dataset,
)
from csng_invariances.metrics_statistics.select_neurons import select_neurons, score
from csng_invariances.data._data_helpers import load_configs

#%%
def load_meis(path: str, device: str = None) -> dict:
    """Loads meis to dict.

    Args:
        path (str): path of directory in which files are stored
        device (str, optional): torch device, if None, tries cuda. Defaults to None.

    Returns:
        dict: key: neuron_idx, value: mei tensor.
    """
    path = Path(path)
    meis = {}
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for file in track(
        path.iterdir(), total=len(list(path.glob("*.npy"))), description="Loading MEIs:"
    ):
        if file.suffix == ".npy":
            _, a = file.name.rsplit("_", 1)
            neuron_idx, _ = a.split(".")
            data = torch.from_numpy(load(file)).to(device)
            meis[neuron_idx] = data
    return meis


#%%
def mei_generation():
    # TODO finalize mei computation

    model_directory = "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03"
    encoding_model = load_encoding_model(model_directory)
    configs = load_configs(model_directory)
    ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
    images, responses = ds.get_dataset()
    criterion = SelectedNeuronActivation()
    gwni = gaussian_white_noise_image(
        size=(1, images.shape[1], images.shape[2], images.shape[3])
    )
    dnn_single_neuron_correlations = dnn_corrs(
        encoding_model,
        images,
        responses,
        batch_size=configs["dataset_config"]["batch_size"],
    )
    linear_filter_single_neuron_correlations = load_single_neuron_correlations_linear_filter(
        "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-30_15:15:09/Correlations.csv"
    )
    selection_score = score(
        dnn_single_neuron_correlations, linear_filter_single_neuron_correlations
    )
    select_neuron_idx = select_neurons(selection_score, 5)
    meis = mei(criterion, encoding_model, gwni, select_neuron_idx, epochs=2000)


def load_mei(path: str, device: str = None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    meis = {}
    for counter, file in enumerate(path.iterdir()):
        mei = torch.from_numpy(load(file))
        mei = mei.to(device)
        meis[counter] = mei
    print(meis)
    return meis


if __name__ == "__main__":
    load_mei("/home/leon/csng_invariances/data/processed/MEIs/2021-12-02_15:46:31")
