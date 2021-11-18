"""Testing scipt"""
#%%
# from os import wait
# from rich import print
# import torch

# from csng_invariances.encoding import *

# from csng_invariances.models.generator import *
# from csng_invariances.models.gan import *

# from csng_invariances.training.generator import NaiveTrainer

# %%
# batch_size = 64
# batches = 100
# epochs = 10

# # %%
# device = "cuda" if torch.cuda.is_available() else "cpu"
# encoding_model = load_encoding_model(
#     "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
# )
# # %%
# # i dont feel like loading data, so here is fake data of the right dimension.
# images = torch.randn(size=(batch_size, 1, 36, 64), device=device, dtype=torch.float)
# responses = torch.randn(size=(batch_size, 5335), device=device, dtype=torch.float)
# # Config
# config = {
#     "latent_space_dimension": 128,
#     "batch_size": batch_size,
#     "layer_growth": 2,
#     "device": device,
#     "batches": batches,
#     "epochs": epochs,
# }
# # %%
# generator_model = GrowingLinearGenerator(images=images, responses=responses, **config)
# # %%
# data = [generator_model.sample_from_normal() for _ in range(batches)]
# # %%
# trainer = NaiveTrainer(
#     generator_model=generator_model, encoding_model=encoding_model, data=data, **config
# )
# for i in range(5):
#     print(
#         f"\n"
#         f"===================================================================\n"
#         f"==================Training Generator for Neuron {i}==================\n"
#         f"===================================================================\n"
#         f"\n"
#     )
#     trainer.train(i)
# %%
################## work on mask ##################
# %%
from typing import Tuple
from rich import print
from csng_invariances.data.datasets import Lurz2021Dataset
from pathlib import Path
import torch

# %%
seed = 1
cuda = True
batch_size = 32
lurz_data_directory = Path.cwd() / "data" / "external" / "lurz2020"
lurz_model_directory = Path.cwd() / "models" / "external" / "lurz2020"
lurz_model_path = lurz_model_directory / "transfer_model.pth.tar"
dataset_config = {
    "paths": [str(lurz_data_directory / "static20457-5-9-preproc0")],
    "batch_size": batch_size,
    "seed": seed,
    "cuda": cuda,
    "normalize": True,
    "exclude": "images",
}
ds = Lurz2021Dataset(dataset_config)
images, responses = ds.get_dataset()
# %%
from csng_invariances.encoding import load_encoding_model

print("loading encoding model")
encoding_model = load_encoding_model(
    "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
)
# %%
# select one image for mask analyzis
image = images[0, :, :, :].reshape(1, 1, images.shape[2], images.shape[3])
response = responses[0, :]
activation = encoding_model(image)

# %%
import numpy as np
from rich.progress import track
from typing import Tuple


def compute_mask(
    image: torch.Tensor,
    response: torch.Tensor,
    encoding_model: torch.nn.Module,
    num_different_pixels: int = 20,
    threshold: float = 0.02,
) -> Tuple:
    """Compute a binary mask of pixels not influencing neural activation.

    Present stimuli to the encoding model and compute activations.
    After that, for each pixel, we carry out the following steps:
        - For every image we produce its copy with a pixel changed to a specific
          value from some test range.
        - Every processed image is then presented to the encoding model.
        - For each pixel, we compute activation for different pixel values across
          the test range. By subtracting the original activation, we can measure
          the standard deviation of these differences. (compare Kovacs 2021)

    Args:
        image (torch.Tensor): image tensor.
        response (torch.Tensor): response tensor.
        encoding_model (torch.nn.Module): encoding model.
        num_different_pixels (int, optional): number of different pixel values
            to use for computation of standard deviation. Defaults to 20.
        threshold (float, optional): Binary threshold value for masking.
            Defaults to 0.02.

    Returns:
        Tuple: Tuple of binary mask tensor and and pixel standard deviation
            tensor. Both of shape (num_neurons, height, width).
    """
    pixel_standard_deviations = torch.empty(
        size=(response.shape[0], image.shape[2], image.shape[3]), device="cuda"
    )
    with torch.no_grad():
        for i in track(range(image.shape[2])):
            for j in range(image.shape[3]):
                activation_differences = torch.empty(
                    size=(response.shape[0], num_different_pixels)
                )
                for counter, value in enumerate(
                    np.linspace(0, 255, num_different_pixels)
                ):
                    img_copy = image
                    img_copy[:, :, i, j] = value
                    output = encoding_model(img_copy)
                    activation_differences[:, counter] = (output - activation).squeeze()
                    del img_copy, output
                pixel_standard_deviations[:, i, j] = torch.std(
                    activation_differences, dim=1
                )
                del activation_differences
        mask = pixel_standard_deviations.ge(0.02)
    return mask, pixel_standard_deviations


mask, _ = compute_mask(image, response, encoding_model)
# %%
import matplotlib.pyplot as plt

for i in range(10):
    print(f"Max. value for this neuron is: {mask[i, :, :].max():.02f}")
    plt.imshow(mask[i, :, :].cpu())
    plt.colorbar()
    plt.show()
# %%
one_mask = mask[1, :, :].reshape(1, 1, mask.shape[1], mask.shape[2])
# %%
one_mask = one_mask.int()
one_mask.max()
# %%
imgs = images[0:64, :, :, :] * one_mask
# %%
images[0, :, :, :]
#%%
for i in range(5):
    plt.imshow(images[i, :, :, :].cpu().squeeze())
    plt.colorbar()
    plt.show()
# %%
