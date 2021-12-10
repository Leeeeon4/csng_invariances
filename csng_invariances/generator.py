 """Generator training experiment module."""

import torch
from csng_invariances.models.encoding import load_encoding_model
from csng_invariances.metrics_statistics.select_neurons import (
    load_selected_neurons_idxs,
)
from csng_invariances.data.preprocessing import (
    image_preprocessing,
    response_preprocessing,
)
from csng_invariances.data.datasets import normal_latent_vector, uniform_latent_vector
from csng_invariances.layers.mask import NaiveMask
from typing import Tuple
from csng_invariances.models.generator import GrowingLinearGenerator
from csng_invariances.training.generator import NaiveTrainer
import matplotlib.pyplot as plt


def load(
    model_directory: str = "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03",
    selected_neuron_idxs_path: str = "/home/leon/csng_invariances/reports/neuron_selection/2021-11-30_15:18:09/selected_neuron_idxs.npy",
    bin_mask_path: str = "/home/leon/csng_invariances/models/masks/2021-11-30_15:19:09/mask.npy",
) -> Tuple[torch.nn.Module, list, torch.Tensor]:
    """Load experiment input

    Args:
        model_directory (str, optional): Model directory path. Defaults to "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03".
        selected_neuron_idxs_path (str, optional): file path. Defaults to "/home/leon/csng_invariances/reports/neuron_selection/2021-11-30_15:18:09/selected_neuron_idxs.npy".
        bin_mask_path (str, optional): file path. Defaults to "/home/leon/csng_invariances/models/masks/2021-11-30_15:19:09/mask.npy".

    Returns:
        [type]: [description]
    """
    encoding_model = load_encoding_model(model_directory)
    selected_neuron_idxs = load_selected_neurons_idxs(selected_neuron_idxs_path)
    bin_mask = NaiveMask.load_binary_mask(bin_mask_path)
    return encoding_model, selected_neuron_idxs, bin_mask


def experiment_training_generator(
    encoding_model: torch.nn.Module,
    selected_neuron_idxs: list,
    bin_mask: torch.Tensor,
    num_batches: int = 15_625,
    latent_space_dimension: int = 128,
    batch_size: int = 64,
) -> torch.nn.Module:

    data = [
        normal_latent_vector(batch_size, latent_space_dimension)
        for _ in range(num_batches)
    ]  # list of <num_batches> latentspace vectors of size (<batch_size>,<latent_space_dimension>)
    show_image = True

    for neuron in selected_neuron_idxs:
        generator_model = GrowingLinearGenerator(layer_growth=2)
        generator_trainer = NaiveTrainer(
            generator_model=generator_model,
            encoding_model=encoding_model,
            data=data,
            mask=bin_mask,
            image_preprocessing=image_preprocessing,
            response_preprocessing=response_preprocessing,
            epochs=20,
            layer_growth=2,
        )
        generator_trainer.train(neuron)
        if show_image:
            eval_sample = uniform_latent_vector(1, 128)
            sample_image = generator_model(eval_sample)
            mask = NaiveMask(bin_mask, neuron)
            masked_image = mask(sample_image)
            masked_image = masked_image.detach().cpu().numpy().squeeze()
            plt.imshow(masked_image)
            plt.savefig(f"test_{neuron}_with_mask.png")
            plt.close()
    return generator_model


from datetime import timedelta
from rich.progress import track
from rich import print
from numpy import save
from pathlib import Path
import torch
from csng_invariances.layers.loss_function import (
    SelectedNeuronActivation,
    SelectedNeuronActivityWithDiffernceInImage,
)
from csng_invariances.layers.mask import NaiveMask
from csng_invariances.mei import load_meis
from csng_invariances.metrics_statistics.correlations import (
    load_single_neuron_correlations_encoding_model,
    load_single_neuron_correlations_linear_filter,
)
from csng_invariances.models.encoding import load_encoding_model
from csng_invariances.metrics_statistics.select_neurons import (
    load_selected_neurons_idxs,
    load_score,
)
from csng_invariances.models.linear_filter import load_linear_filter
from csng_invariances.data.datasets import normal_latent_vector, uniform_latent_vector
from csng_invariances.models.generator import (
    FullyConnectedGeneratorWithGaussianBlurring,
    GrowingLinearGenerator,
    FullyConnectedGenerator,
)
from csng_invariances.training.generator import NaiveTrainer
from csng_invariances.data.preprocessing import (
    image_preprocessing,
    response_preprocessing,
)
from csng_invariances._utils.utlis import string_time
from csng_invariances.metrics_statistics.clustering import cluster_generated_images
from csng_invariances._utils.plotting import (
    plot_examples_of_generated_images,
    plot_neuron_x_with_8_clusters,
)

#%%


# %%
########################User setup###################################
batch_size = 64
latent_space_dimension = 128
num_training_batches = 16  # 15625
num_generation_batches = 4  # 1563
image_shape = (batch_size, 1, 36, 64)
masked = False  # Applies mask during training and generation
generate = True  # generates images after training
show_image = True  # save first image of each batch after training as *.png
selected_neuron_idxs_file = "/home/leon/csng_invariances/reports/neuron_selection/2021-11-30_15:18:09/selected_neuron_idxs.npy"
encoding_model_directory = (
    "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03"
)
bin_mask_file = "/home/leon/csng_invariances/models/masks/2021-11-30_15:19:09/mask.npy"
roi_file = "/home/leon/csng_invariances/data/processed/roi/2021-11-29_15:52:35/pixel_standard_deviation.npy"
linear_filter_file = "/home/leon/csng_invariances/data/processed/linear_filter/2021-11-30_15:15:09/evaluated_filter.npy"
meis_directory = "/home/leon/csng_invariances/data/processed/MEIs/2021-12-02_15:46:31"
score_file = "/home/leon/csng_invariances/reports/scores/2021-12-02_15:46:31/score.npy"
lrf_correlation_file = "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-30_15:15:09/Correlations.csv"
enc_correlation_file = "/home/leon/csng_invariances/reports/encoding/single_neuron_correlations/2021-12-02_15:46:31/single_neuron_correlations.npy"
# %%
###########################load model, mask and selected neurons###############
selected_neuron_idxs = load_selected_neurons_idxs(selected_neuron_idxs_file)
encoding_model = load_encoding_model(encoding_model_directory)
bin_mask = NaiveMask.load_binary_mask(bin_mask_file)
lrf = load_linear_filter(linear_filter_file)
meis = load_meis(meis_directory)
roi = NaiveMask.load_pixel_standard_deviations(roi_file)
roi = roi.reshape(roi.shape[0], 1, roi.shape[1], roi.shape[2])
score = load_score(score_file)
lrf_correlations = load_single_neuron_correlations_linear_filter(lrf_correlation_file)
enc_correlations = load_single_neuron_correlations_encoding_model(enc_correlation_file)
#%%
##########################generate training data###############################
data = [
    normal_latent_vector(batch_size, latent_space_dimension)
    for _ in range(num_training_batches)
]
eval_samples = [
    uniform_latent_vector(batch_size, latent_space_dimension)
    for _ in range(num_generation_batches)
]

print(
    f"Sum of difference of two normal vector batches: {(data[0]-data[1]).sum()}\n"
    f"Sum of difference of two uniform vector batches: "
    f"{(eval_samples[0]-eval_samples[1]).sum()}"
)
# %%
#######################RUN EXPERIMENT#############################################
from time import perf_counter

def generator_training_and_generation():
    start = perf_counter()
    print(f"Masking is {masked}.")
    if masked:
        m = "masked"
        mask = bin_mask
    else:
        m = "not_masked"
        mask = None
    t = string_time()
    config = {}
    config["Timestamp"] = t
    intermediates = {}
    for neuron_counter, neuron in enumerate(selected_neuron_idxs):
        # Print current Neuron
        print(f"Neuron {neuron_counter+1} / {len(selected_neuron_idxs)}")

        # Create generator model
        generator_model = FullyConnectedGeneratorWithGaussianBlurring(
            output_shape=image_shape,
            latent_space_dimension=latent_space_dimension,
            sigma=0.8,  # sigma 0.5 still had artifact, sigma 1 not
            batch_norm=True,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(
            params=generator_model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        )

        # Create loss function
        loss_function = SelectedNeuronActivityWithDiffernceInImage(0.0001)

        # Create trainer
        generator_trainer = NaiveTrainer(
            generator_model=generator_model,
            encoding_model=encoding_model,
            optimizer=optimizer,
            loss_function=loss_function,
            data=data,
            mask=mask,
            image_preprocessing=image_preprocessing,
            response_preprocessing=response_preprocessing,
            epochs=20,
            weight_decay=0.1,
            show_development=True,
            prep_video=False,
            config=config,
        )

        generator_model, epochs_images, config = generator_trainer.train(neuron)
        t = config["Timestamp"] + "_" + config["wandb_name"]
        if generate:
            data_directory = (
                Path.cwd()
                / "data"
                / "processed"
                / "generator"
                / "after_training"
                / t
                / f"neuron_{neuron}"
            )
            data_directory.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():  # gradient not needed, increases speed
                generated_images = torch.empty(
                    size=(
                        batch_size * num_training_batches,
                        image_shape[1],
                        image_shape[2],
                        image_shape[3],
                    ),
                    device="cuda",
                )
                tensors = []
                for batch_counter, eval_sample in track(
                    enumerate(eval_samples),
                    total=len(eval_samples),
                    description=f"Generating images: ",
                ):

                    # eval_sample is tensor of shape (batch_size, latent_vector_dimension)
                    sample_image = generator_model(eval_sample)

                    # sample_image is tensor of shape (batch_size, 1, height, width)
                    if masked:
                        masking = NaiveMask(mask, neuron)
                        masked_image = masking(sample_image)
                    else:
                        masked_image = sample_image

                    # preprocessed_image is N(0,1) normalized and scaled to be [0,1]
                    preprocessed_image = image_preprocessing(masked_image)

                    # plot examples if True
                    if show_image:
                        image_directory = (
                            Path.cwd()
                            / "reports"
                            / "figures"
                            / "generator"
                            / "after_training"
                            / t
                            / f"neuron_{neuron}"
                        )
                        image_directory.mkdir(parents=True, exist_ok=True)
                        activation = encoding_model(preprocessed_image)
                        plot_examples_of_generated_images(
                            selected_neuron_idx=neuron,
                            batch_counter=batch_counter,
                            sample_image=sample_image,
                            masked_image=masked_image,
                            preprocessed_image=preprocessed_image,
                            image_directory=image_directory,
                            encoding_model=encoding_model,
                        )

                    # add into one vector
                    tensors.append(preprocessed_image)
                generated_images = torch.cat(tensors, dim=0)

                # compute activations and save data
                activations = encoding_model(generated_images)
                activations = activations[:, neuron]
                file_name = f"generated_images.npy"
                save(
                    file=data_directory / file_name,
                    arr=generated_images.detach().cpu().numpy(),
                )
                file_name = f"activations.npy"
                save(
                    file=data_directory / file_name,
                    arr=activations.detach().cpu().numpy(),
                )

                # cluster images (and activations accordingly) to detect different images
                clustered_images, clustered_activations = cluster_generated_images(
                    generated_images, activations, neuron, show=False
                )

                # save clusters
                cluster_directory = data_directory / "clustered"
                cluster_directory.mkdir(parents=True, exist_ok=True)
                for cluster_counter, (image_cluster, activation_cluster) in enumerate(
                    zip(clustered_images, clustered_activations)
                ):
                    save(
                        file=cluster_directory / f"images_cluster_{cluster_counter}.npy",
                        arr=image_cluster.detach().cpu().numpy(),
                    )
                    save(
                        file=cluster_directory
                        / f"activations_cluster_{cluster_counter}.npy",
                        arr=activation_cluster.detach().cpu().numpy(),
                    )

                # print(clustered_activations)
                plot_neuron_x_with_8_clusters(
                    selected_neuron_idx=neuron,
                    lrf=lrf,
                    meis=meis,
                    roi=roi,
                    mask=mask,
                    epochs_images=epochs_images,
                    clustered_images=clustered_images,
                    clustered_activations=clustered_activations,
                    score=score,
                    lrf_correlations=lrf_correlations,
                    enc_correlations=enc_correlations,
                    training_data=data,
                    generation_data=eval_samples,
                    encoding_model=encoding_model,
                    generator_model=generator_model,
                    config=config,
                )

        intermediate = perf_counter()
        intermediates[neuron_counter] = intermediate
        if neuron_counter == 0:

            print(f"Current neuron {neuron} took: {(intermediate-start):.2f}s")
        else:
            print(
                f"Current neuron {neuron} took: {(intermediates[neuron_counter]-intermediates[neuron_counter-1]):.2f}s"
            )
    end = perf_counter()
    print(f"Complete process took {timedelta(seconds=(end-start))}")


if __name__ == "__main__":
    # = load()
    pass
