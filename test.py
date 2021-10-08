import numpy as np
from rich import print
from csng_invariances.linear_receptive_field import *


if __name__ == "__main__":

    # Regularization factors
    reg_factors = [1 * 10 ** x for x in np.linspace(-5, 5, 50)]

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
