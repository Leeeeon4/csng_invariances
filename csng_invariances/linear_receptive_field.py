"""Module providing functions for conducting linear receptive field estimate experiments.
"""

import numpy as np
from pathlib import Path
import torch
from rich import print
import csng_invariances.datasets.antolik2016 as al
import csng_invariances.datasets.lurz2020 as lu
import csng_invariances.utility.lin_filter as lin_fil
from csng_invariances.utility.data_helpers import normalize_tensor_to_0_1 as norm_0_1


# Lurz
def get_lurz_dataset():
    """Get Lurz data.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses.
    """
    # Load dataset
    experiment_path = str(
        Path.cwd() / "data" / "external" / "lurz2020" / "static20457-5-9-preproc0"
    )
    print(f"Loading dataset from {experiment_path}.")
    dataloaders = lu.get_dataloaders()
    train_images, train_responses = lu.get_complete_dataset(dataloaders, "train")
    val_images, val_responses = lu.get_complete_dataset(dataloaders, "validation")
    return train_images, train_responses, val_images, val_responses


# Antolik
def get_antolik_dataset(region):
    """Get Antolik data.

    Args:
        region (str): Region to examine.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses
    """
    # Load dataset
    experiment_path = Path.cwd() / "data" / "external" / "antolik2016" / "Data"
    print(f"Loading data from {experiment_path}.")
    dataloaders = al.get_dataloaders()
    train_images, train_responses = al.get_complete_dataset(
        dataloaders, "train", region
    )
    val_images, val_responses = al.get_complete_dataset(
        dataloaders, "validation", region
    )
    return train_images, train_responses, val_images, val_responses


def globally_regularized_linear_receptive_field(
    reg_factors, train_images, train_responses, val_images, val_responses
):
    """Globally regularized linear receptive field estimate experiments on Lurz data.

    Conduct hyperparametersearch for regularization factor.
    Report linear filter and single neuron correlations on validation data.

    Args:
        reg_factors (list): List of regularization factors for hyperparametersearch.
        train_images (np.array): 2D representation of train image data. Images
            are flattened.
        train_responses (np.array): 2D representation of train response data.
        val_images (np.array): 2D representation of validation image data. Images
            are flattened.
        val_responses (np.array): 2D representation of val response data.

    Returns:
        tuple: tuple of dict of regularization factors and correlations,filter
            and dictionary of neurons and single neuron correlation.
    """

    # Build filters for estimation of linear receptive field
    TrainFilter = lin_fil.GlobalRegularizationFilter(
        norm_0_1(train_images), norm_0_1(train_responses)
    )
    ValFilter = lin_fil.GlobalRegularizationFilter(
        norm_0_1(val_images), norm_0_1(val_responses)
    )

    # hyperparametersearch
    print(
        f"Conducting hyperparametersearch for regularization factor from:\n{reg_factors}."
    )
    Hyperparametersearch = lin_fil.GlobalHyperparametersearch(
        TrainFilter, ValFilter, reg_factors
    )
    Hyperparametersearch.conduct_search()
    Hyperparametersearch.compute_best_parameter()
    parameters = Hyperparametersearch.get_parameters()

    fil = TrainFilter.train(parameters)

    # report linear filter
    ValFilter.evaluate(fil=fil)


def individually_regularized_linear_receptive_field(
    reg_factors, train_images, train_responses, val_images, val_responses
):
    """Individually regularized linear receptive field estimate experiments on Lurz data.

    Conduct hyperparametersearch for regularization factor.
    Report linear filter and single neuron correlations on validation data.

    Args:
        reg_factors (list): List of regularization factors for hyperparametersearch.
        train_images (np.array): 2D representation of train image data. Images
            are flattened.
        train_responses (np.array): 2D representation of train response data.
        val_images (np.array): 2D representation of validation image data. Images
            are flattened.
        val_responses (np.array): 2D representation of val response data.

    Returns:
        tuple: tuple of 2D array of neuron, regularization factor and single neuron
            correlation, filter and dictionary of neurons and single neuron
            correlation.
    """
    # Build filters for estimation of linear receptive field
    TrainFilter = lin_fil.IndividualRegularizationFilter(
        norm_0_1(train_images), norm_0_1(train_responses)
    )
    ValFilter = lin_fil.IndividualRegularizationFilter(
        norm_0_1(val_images), norm_0_1(val_responses)
    )

    # hyperparametersearch
    print(
        f"Conducting hyperparametersearch for regularization factor from:\n{reg_factors}"
    )
    Hyperparametersearch = lin_fil.IndividualHyperparametersearch(
        TrainFilter, ValFilter, reg_factors
    )
    Hyperparametersearch.conduct_search()
    Hyperparametersearch.compute_best_parameter()
    parameters = Hyperparametersearch.get_parameters()
    fil = TrainFilter.train(parameters)

    # report linear filter
    ValFilter.evaluate(fil=fil)


if __name__ == "__main__":
    pass
