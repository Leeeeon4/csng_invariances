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


if __name__ == "__main__":
    # = load()
    pass
