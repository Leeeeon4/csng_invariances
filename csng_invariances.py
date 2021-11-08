"""Main module containing the experiments that can be run."""

import numpy as np
import matplotlib as mpl

from rich import print

from csng_invariances.linear_receptive_field import *


def linear_receptive_field_experiments():
    """Run linear receptive field experiment.

    For the Lurz dataset the first a hyperparametersearch, then a filter
    evaluation based on the hyper parametersearch is conducted when global
    regulariztaion is used, then when single neuron regularization is used.
    For the Antolik dataset this is done for each of the three regions.
    """
    ########################## Regularization factors #########################
    reg_factors = [1 * 10 ** x for x in np.linspace(-5, 5, 50)]

    ################################## Lurz ###################################

    ################################## Data ###################################
    print("\n\n================================================")
    print("Begin linear receptive field estimation experiment on Lurz dataset.")
    print("================================================\n\n")
    (
        train_images,
        train_responses,
        val_images,
        val_responses,
    ) = get_lurz_dataset()
    print("-----------------------------------------------\n")

    ########################## Global regularization ##########################
    print("\n-----------------------------------------------")
    print("Begin globally regularized linear receptive field estimate experiments.")
    print("-----------------------------------------------\n")

    # Conduct hyperparameter search
    globally_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )

    # Create directory and store filters
    print("\n-----------------------------------------------")
    print("Finished globally regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")

    # ######################## Individual regularization ########################
    print("\n-----------------------------------------------")
    print("Begin indivially regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")

    # Conduct hyperparameter search
    individually_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )
    print("\n\n================================================")
    print("Lurz dataset concluded.")
    print("================================================\n\n")

    ################################# Antolik #################################

    print("\n\n================================================")
    print("Begin Antolik dataset.")
    print("================================================\n\n")
    for region in ["region1", "region2", "region3"]:
        ############################# Region ##################################
        print("\n-----------------------------------------------")
        print(f"Being {region}.")
        print("-----------------------------------------------\n")

        ################################# Data ################################
        (
            train_images,
            train_responses,
            val_images,
            val_responses,
        ) = get_antolik_dataset(region)

        ######################## Global regularization ########################
        print("Begin globally regularized linear receptive field estimate experiments.")
        print("-----------------------------------------------\n")

        # Conduct hyper parameter search
        globally_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )
        print(
            "Finished globally regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")

        ###################### Individual regularization ######################
        print(
            "Begin indivially regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")

        # Conduct hyperparametersearch
        individually_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )
        print("\n-----------------------------------------------")
        print(f"Conclude {region}.")

    print("\n\n================================================")
    print("Antolik dataset concluded.\n\n")
    print("================================================\n\n")


if __name__ == "__main__":
    """Concept of experiments.

    1. Compute Predictions and single neuron correlations:
        - discriminator Predictions - r_CNN:
            possibly train discriminator (encoding model) on data
        - linear Predictions - r_RF:
            (X.T*X + lambda*L)^-1 * X.T * y
            X: images
            lambda: regularization factor
            L: regularization
            y: responses
    3. Compute score
        (1 - r_RF)*r_CNN
    4. Compute Most Exciting Image:
        - gradient ascent:
            - apply gradient on image
                -  initial image is gaussian white noise
            - gaussian blur application after each gradient ascent step with
              decreasing standard deviation
            - low pass filtering - in fourier domain - of gradient before application
    5. Train generator:
        - train generator model on: n(0,1)
        - optimize neural activation of selected neurons on MEI (target)
    6. Generate Samples:
        - from: u(-2,2)
    7. Compute and apply ROI
    8. Cluster 'most representative/most different' samples
    9. Analyze important samples
    """
    mpl.rc_file(Path.cwd() / "matplotlibrc")
    linear_receptive_field_experiments()
