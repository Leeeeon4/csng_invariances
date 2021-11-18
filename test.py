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
from rich import print
from csng_invariances.data.datasets import Lurz2021Dataset
from pathlib import Path
import torch

# %%
def print_cuda():
    res = torch.cuda.memory_reserved(0)
    allo = torch.cuda.memory_allocated(0)
    print(
        f"Reserved memory: {res:,}\n"
        f"Allocated memory: {allo:,}\n"
        f"Free memory: {(res-allo):,}"
    )


# %%
print(
    f"Total amount of GPU memory: {torch.cuda.get_device_properties(0).total_memory:,}"
)
print_cuda()
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
print("loading data")
ds = Lurz2021Dataset(dataset_config)
images, responses = ds.get_dataset()
print_cuda()
# %%
from csng_invariances.encoding import load_encoding_model

print("loading encoding model")
encoding_model = load_encoding_model(
    "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
)
print_cuda()
# %%
# select one image for mask analyzis
print("select image\ncould free mem here")
image = images[0, :, :, :].reshape(1, 1, images.shape[2], images.shape[3])
response = responses[0, :]
activation = encoding_model(image)
# activation = activation.cpu()
del images, responses
print_cuda()
# %%
activation
# %%
import numpy as np
from rich.progress import track

num_different_pixels = 20
stds = torch.empty(
    size=(response.shape[0], image.shape[2], image.shape[3]), device="cuda"
)
with torch.no_grad():
    for i in track(range(image.shape[2])):
        for j in range(image.shape[3]):
            activation_differences = torch.empty(
                size=(response.shape[0], num_different_pixels)
            )
            for counter, value in enumerate(np.linspace(0, 255, num_different_pixels)):
                img_copy = image
                img_copy[:, :, i, j] = value
                output = encoding_model(img_copy)
                activation_differences[:, counter] = (output - activation).squeeze()
                del img_copy, output
            stds[:, i, j] = torch.std(activation_differences, dim=1)
            del activation_differences

# %%
print(stds.shape)
# %%
print_cuda()
# %%
