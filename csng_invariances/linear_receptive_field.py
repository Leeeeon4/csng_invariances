import numpy as np
from pathlib import Path
import torch
from rich import print
import datasets.antolik2016 as al
import datasets.lurz2020 as lu
import utility.lin_filter as lin_fil
from utility.data_helpers import normalize_tensor_to_0_1 as norm_0_1


# Testdata
def get_test_dataset():
    """Return small dataset for testing.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses.
    """
    train_images = torch.from_numpy(np.random.randn(100, 1, 12, 13))
    train_responses = torch.from_numpy(np.random.randn(100, 14))
    val_images = torch.from_numpy(np.random.randn(20, 1, 12, 13))
    val_responses = torch.from_numpy(np.random.randn(20, 14))
    return train_images, train_responses, val_images, val_responses


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
    dataloaders = lu.static_loaders(
        **{
            "paths": [experiment_path],
            "batch_size": 64,
            "seed": 1,
            "cuda": False,
            "normalize": True,
            "exclude": "images",
        }
    )
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
        tuple: tuple of filter and dictionary of neurons and single neuron correlation.
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
    parameter, _ = lin_fil.conduct_global_hyperparametersearch(
        TrainFilter, ValFilter, reg_factors
    )
    fil = TrainFilter.train(parameter)

    # report linear filter
    neural_correlations = ValFilter.evaluate(fil=fil, output=True)

    return fil, neural_correlations


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
        tuple: tuple of filter and dictionary of neurons and single neuron correlation.
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
    hyperparametersearch = lin_fil.conduct_individual_hyperparametersearch(
        TrainFilter, ValFilter, reg_factors
    )
    regularization_factors = hyperparametersearch[:, 1]
    fil = TrainFilter.train(regularization_factors)

    # report linear filter
    neural_correlations = ValFilter.evaluate(fil=fil, output=False)

    return fil, neural_correlations


if __name__ == "__main__":
    # Regularization factors
    reg_factors = [1 * 10 ** x for x in np.linspace(-1, 4, 10)]

    # Lurz
    # Data
    print("\n\n================================================")
    print("Begin linear receptive field estimation experiment on Lurz dataset.")
    print("================================================\n\n")
    (
        train_images,
        train_responses,
        val_images,
        val_responses,
    ) = get_lurz_dataset()
    print(
        f"{train_images.shape}\n{train_responses.shape}\n{val_images.shape}\n{val_responses.shape}"
    )
    print("-----------------------------------------------\n")

    # testdata
    train_images, train_responses, val_images, val_responses = get_test_dataset()

    # Global regularization
    print("\n-----------------------------------------------")
    print("Begin globally regularized linear receptive field estimate experiments.")
    print("-----------------------------------------------\n")
    globally_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )
    print("\n-----------------------------------------------")
    print("Finished globally regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")

    # Individual regularization
    print("\n-----------------------------------------------")
    print("Begin indivially regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")
    individually_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )
    print("\n\n================================================")
    print("Lurz dataset concluded.")
    print("================================================\n\n")

    print("\n\n================================================")
    print("Begin Antolik dataset.")
    print("================================================\n\n")

    # Antolik
    for region in ["region1", "region2", "region3"]:
        print("\n-----------------------------------------------")
        print(f"Being {region}.")
        print("-----------------------------------------------\n")

        # # Data
        (
            train_images,
            train_responses,
            val_images,
            val_responses,
        ) = get_antolik_dataset(region)
        print(
            f"{train_images.shape}\n{train_responses.shape}\n{val_images.shape}\n{val_responses.shape}"
        )
        # Testdata
        train_images, train_responses, val_images, val_responses = get_test_dataset()

        # Global regularization
        print("Begin globally regularized linear receptive field estimate experiments.")
        print("-----------------------------------------------\n")
        globally_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )
        print(
            "Finished globally regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")

        # Individual regularization
        print(
            "Begin indivially regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")
        individually_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )
        print("\n-----------------------------------------------")
        print(f"Conclude {region}.")

    print("\n\n================================================")
    print("Antolik dataset concluded.\n\n")
    print("================================================\n\n")
    # pass
