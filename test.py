"""Testing scipt"""
#%%
from os import wait
from rich import print
import torch

# testcomment
from csng_invariances.encoding import *

from csng_invariances.models.generator import *
from csng_invariances.models.gan import *

from csng_invariances.training.generator import NaiveTrainer

# %%
batch_size = 64
batches = 100
epochs = 10

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
encoding_model = load_encoding_model(
    "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
)
# %%
# i dont feel like loading data, so here is fake data of the right dimension.
images = torch.randn(size=(batch_size, 1, 36, 64), device=device, dtype=torch.float)
responses = torch.randn(size=(batch_size, 5335), device=device, dtype=torch.float)
# Config
config = {
    "latent_space_dimension": 128,
    "batch_size": batch_size,
    "layer_growth": 2,
    "device": device,
    "batches": batches,
    "epochs": epochs,
}
# %%
generator_model = GrowingLinearGenerator(images=images, responses=responses, **config)
# %%
data = [generator_model.sample_from_normal() for _ in range(batches)]
# %%
trainer = NaiveTrainer(
    generator_model=generator_model, encoding_model=encoding_model, data=data, **config
)
for i in range(5):
    print(
        f"\n"
        f"===================================================================\n"
        f"==================Training Generator for Neuron {i}==================\n"
        f"===================================================================\n"
        f"\n"
    )
    trainer.train(i)
# %%
