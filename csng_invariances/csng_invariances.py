"""Main module containing the experiments that can be run."""

import numpy as np
import matplotlib as mpl
from rich import print
from linear_receptive_field import *


def linear_receptive_field_experiments():
    """Run linear receptive field experiment.

    For the Lurz dataset the first a hyperparametersearch, then a filter
    evaluation based on the hyper parametersearch is conducted when global
    regulariztaion is used, then when single neuron regularization is used.
    For the Antolik dataset this is done for each of the three regions.
    """
    ########################## Regularization factors #########################
<<<<<<< HEAD
<<<<<<< HEAD
    reg_factors = [1 * 10 ** x for x in np.linspace(-1, 5, 1)]
=======
    reg_factors = [1 * 10 ** x for x in np.linspace(-5, 5, 50)]
>>>>>>> e69bf1503673fd173a6ae273fb835fc7625eea48
=======
    reg_factors = [1 * 10 ** x for x in np.linspace(-5, 5, 50)]
>>>>>>> e69bf1503673fd173a6ae273fb835fc7625eea48

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
    # print("\n-----------------------------------------------")
    # print("Begin indivially regularized linear receptive field estimate experiment.")
    # print("-----------------------------------------------\n")

    # # Conduct hyperparameter search
    # individually_regularized_linear_receptive_field(
    #     reg_factors, train_images, train_responses, val_images, val_responses
    # )
    # print("\n\n================================================")
    # print("Lurz dataset concluded.")
    # print("================================================\n\n")

<<<<<<< HEAD
<<<<<<< HEAD
    # ################################# Antolik #################################

    # print("\n\n================================================")
    # print("Begin Antolik dataset.")
    # print("================================================\n\n")
    # for region in ["region1", "region2", "region3"]:
    #     ############################# Region ##################################
    #     print("\n-----------------------------------------------")
    #     print(f"Being {region}.")
    #     print("-----------------------------------------------\n")

    #     ################################# Data ################################
    #     (
    #         train_images,
    #         train_responses,
    #         val_images,
    #         val_responses,
    #     ) = get_antolik_dataset(region)

    #     ######################## Global regularization ########################
    #     print("Begin globally regularized linear receptive field estimate experiments.")
    #     print("-----------------------------------------------\n")

    #     # Conduct hyper parameter search
    #     globally_regularized_linear_receptive_field(
    #         reg_factors, train_images, train_responses, val_images, val_responses
    #     )
    #     print(
    #         "Finished globally regularized linear receptive field estimate experiment."
    #     )
    #     print("-----------------------------------------------\n")

    #     ###################### Individual regularization ######################
    #     print(
    #         "Begin indivially regularized linear receptive field estimate experiment."
    #     )
    #     print("-----------------------------------------------\n")

    #     # Conduct hyperparametersearch
    #     individually_regularized_linear_receptive_field(
    #         reg_factors, train_images, train_responses, val_images, val_responses
    #     )
    #     print("\n-----------------------------------------------")
    #     print(f"Conclude {region}.")

    # print("\n\n================================================")
=======
=======
>>>>>>> e69bf1503673fd173a6ae273fb835fc7625eea48
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

        # ###################### Individual regularization ######################
        # print(
        #     "Begin indivially regularized linear receptive field estimate experiment."
        # )
        # print("-----------------------------------------------\n")

        # # Conduct hyperparametersearch
        # individually_regularized_linear_receptive_field(
        #     reg_factors, train_images, train_responses, val_images, val_responses
        # )
        # print("\n-----------------------------------------------")
        print(f"Conclude {region}.")

    print("\n\n================================================")
>>>>>>> e69bf1503673fd173a6ae273fb835fc7625eea48
    print("Antolik dataset concluded.\n\n")
    print("================================================\n\n")


def matplotlib_style_setup():
    # Setup Matplotlib style based on matplotlibrc-file.
    # If two plot are supposed to fit on one presentation slide, use figsize = figure_sizes["half"]
    mpl.rc_file("matplotlibrc")
    figure_sizes = {
        "full": (8, 5.6),
        "half": (5.4, 3.8),
    }


if __name__ == "__main__":
    """Concept:
    1. Train discriminator of data
    2. Compute Predictions:
    - discriminator Predictions
    - linear Predictions
    3. Compute score
    4. Compute MEI
    5. Compute ROI
    6. Train generator n(0,1), generator would need to genereate coordinates for example images as well?
    7. Generate Samples u(-2,2)
    8. Cluster 'most representative/most different' samples
    9. Analyze important samples
    """
    matplotlib_style_setup()
    linear_receptive_field_experiments()
