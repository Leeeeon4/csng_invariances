"""Module providing functions for conducting linear receptive field estimate experiments.
"""

from pathlib import Path
from rich import print
import datasets.antolik2016 as al
import datasets.lurz2020 as lu
import utility.lin_filter as lin_fil
from utility.data_helpers import normalize_tensor_to_0_1 as norm_0_1
import argparse
from csng_invariances.utility.ipyhandler import automatic_cwd


def get_lurz_dataset():
    """Get Lurz data.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses.
    """
    # Load dataset
    experiment_path = str(
        automatic_cwd() / "data" / "external" / "lurz2020" / "static20457-5-9-preproc0"
    )
    print(f"Loading dataset from {experiment_path}.")
    dataloaders, _ = lu.get_dataloaders()
    train_images, train_responses = lu.get_complete_dataset(dataloaders, "train")
    val_images, val_responses = lu.get_complete_dataset(dataloaders, "validation")
    return train_images, train_responses, val_images, val_responses


def get_antolik_dataset(region):
    """Get Antolik data.

    Args:
        region (str): Region to examine.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses
    """
    # Load dataset
    experiment_path = automatic_cwd() / "data" / "external" / "antolik2016" / "Data"
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
        norm_0_1(train_images), norm_0_1(train_responses), reg_type="ridge regularized"
    )
    ValFilter = lin_fil.GlobalRegularizationFilter(
        norm_0_1(val_images), norm_0_1(val_responses), reg_type="ridge regularized"
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
    ValFilter.evaluate(fil=fil, report_dir=Hyperparametersearch.report_dir)


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
        norm_0_1(train_images), norm_0_1(train_responses), reg_type="ridge regularized"
    )
    ValFilter = lin_fil.IndividualRegularizationFilter(
        norm_0_1(val_images), norm_0_1(val_responses), reg_type="ridge regularized"
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
    ValFilter.evaluate(fil=fil, report_dir=Hyperparametersearch.report_dir)


def linear_receptive_field_argparse(parser):
    """Handle evaluation argparsing.

    Args:
        parser (ArgumentParser): ArgumentParser for function call.

    Returns:
        dict: dictionary of kwargs and values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_path",
        type=str,
        help="Path to the report to be evaluated.",
        default="/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-10-14_18:13:10/hyperparametersearch_report.npy",
    )
    parser.add_argument(
        "--filter_path",
        type=str,
        help="Path to the filter to be evaluated.",
        default="/home/leon/csng_invariances/models/linear_filter/2021-10-14_18:13:10/evaluated_filter.npy",
    )
    parser.add_argument(
        "--count", type=int, help="Number of filters to plot.", default=5
    )
    kwargs = parser.parse_args()
    return vars(kwargs)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # kwargs = linear_receptive_field_argparse()
    # evaluate_reports(**vars(kwargs))
    pass
