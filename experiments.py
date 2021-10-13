import numpy as np
from rich import print
from csng_invariances.linear_receptive_field import *
import datetime


def main():
    ######################### Regularization factors ########################
    reg_factors = [1 * 10 ** x for x in np.linspace(-5, 5, 50)]

    ############################### Lurz ###############################
    ############################### Data ###############################
    print("\n\n================================================")
    print("Begin linear receptive field estimation experiment on Lurz dataset.")
    print("================================================\n\n")

    # real data
    # (
    #     train_images,
    #     train_responses,
    #     val_images,
    #     val_responses,
    # ) = get_lurz_dataset()
    # print(
    #     f"{train_images.shape}\n{train_responses.shape}\n{val_images.shape}\n{val_responses.shape}"
    # )
    # print("-----------------------------------------------\n")

    # test data
    train_images, train_responses, val_images, val_responses = get_test_dataset()

    ####################### Global regularization ######################
    print("\n-----------------------------------------------")
    print("Begin globally regularized linear receptive field estimate experiments.")
    print("-----------------------------------------------\n")

    # Store current time
    t = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    # Conduct hyperparameter search
    fil, neural_correlations = globally_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )

    # Create directory and store filters
    global_reg_lurz = Path.cwd() / "models" / t
    global_reg_lurz.mkdir(parents=True, exist_ok=True)
    global_reg_lurz = global_reg_lurz / "Globally_regularized_linear_filter_lurz.npy"
    np.save(str(global_reg_lurz), fil)
    print("\n-----------------------------------------------")
    print("Finished globally regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")

    ###################### Individual regularization ######################
    print("\n-----------------------------------------------")
    print("Begin indivially regularized linear receptive field estimate experiment.")
    print("-----------------------------------------------\n")

    # Conduct hyperparameter search
    fil, neural_correlations = individually_regularized_linear_receptive_field(
        reg_factors, train_images, train_responses, val_images, val_responses
    )

    # Store filter
    individual_reg_lurz = Path.cwd() / "models" / t
    individual_reg_lurz = (
        individual_reg_lurz / "Individually_regularized_linear_filter_lurz.npy"
    )
    np.save(str(individual_reg_lurz), fil)
    print("\n\n================================================")
    print("Lurz dataset concluded.")
    print("================================================\n\n")

    ############################## Antolik ##############################

    print("\n\n================================================")
    print("Begin Antolik dataset.")
    print("================================================\n\n")
    for region in ["region1", "region2", "region3"]:
        print("\n-----------------------------------------------")
        print(f"Being {region}.")
        print("-----------------------------------------------\n")

        ############################### Data ##############################
        # real data
        # (
        #     train_images,
        #     train_responses,
        #     val_images,
        #     val_responses,
        # ) = get_antolik_dataset(region)
        # print(
        #     f"{train_images.shape}\n{train_responses.shape}\n{val_images.shape}\n{val_responses.shape}"
        # )

        # test data
        train_images, train_responses, val_images, val_responses = get_test_dataset()

        ####################### Global regularization ######################
        print("Begin globally regularized linear receptive field estimate experiments.")
        print("-----------------------------------------------\n")

        # Conduct hyper parameter search
        fil, neural_correlations = globally_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )

        # Store filter
        global_reg_antolik = Path.cwd() / "models" / t / region
        global_reg_antolik = (
            global_reg_antolik / "Globally_regularized_linear_filter_Antolik.npy"
        )
        np.save(str(global_reg_antolik), fil)
        print(
            "Finished globally regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")

        ###################### Individual regularization #####################
        print(
            "Begin indivially regularized linear receptive field estimate experiment."
        )
        print("-----------------------------------------------\n")

        # Conduct hyperparametersearch
        fil, neural_correlations = individually_regularized_linear_receptive_field(
            reg_factors, train_images, train_responses, val_images, val_responses
        )

        # Store filter
        individual_reg_antolik = Path.cwd() / "models" / t / region
        individual_reg_antolik = (
            individual_reg_antolik
            / "Individually_regularized_linear_filter_Antolik.npy"
        )
        np.save(str(individual_reg_antolik), fil)
        print("\n-----------------------------------------------")
        print(f"Conclude {region}.")

    print("\n\n================================================")
    print("Antolik dataset concluded.\n\n")
    print("================================================\n\n")


if __name__ == "__main__":
    main()
