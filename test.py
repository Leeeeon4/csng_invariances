"""Testing scipt"""
#%%
from os import wait
import torch

from csng_invariances.data.preprocessing import image_preprocessing
from csng_invariances.encoding import *

from csng_invariances.models.generator import *
from csng_invariances.models.gan import *

from csng_invariances.training.generator import NaiveTrainer


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
encoding_model = load_encoding_model(
    "/home/leon/csng_invariances/models/encoding/2021-11-04_11:39:48"
)

#%%
batch_size = 64
batches = 1560
epochs = 50
# i dont feel like loading data, so here is fake data of the right dimension.
images = torch.randn(size=(batch_size, 1, 36, 64), device=device, dtype=torch.float)
responses = torch.randn(size=(batch_size, 5335), device=device, dtype=torch.float)
# Config
config = {
    "images": images,
    "responses": responses,
    "latent_space_dimension": 128,
    "batch_size": batch_size,
    "encoding_model": encoding_model,
    "layer_growth": 2,
    "device": device,
    "batches": batches,
    "epochs": epochs,
}
#%%
generator_model = GrowingLinearGenerator(**config)
#%%
data = [generator_model.sample_from_normal() for _ in range(batches)]

trainer = NaiveTrainer(generator_model=generator_model, data=data, **config)
m = trainer.train(5174)
# %%
from tqdm import tqdm
from rich.progress import track
import time
#%%
start = time.time()
for _ in tqdm(range(1 * 10 ** 8)):
    _

tqdm_timer = time.time()
print(f"tqdm took {round((tqdm_timer-start),2)}s")
#%%
start = time.time()
for _ in track(range(1 * 10 ** 8)):
    _

track_timer = time.time()
print(f"track took {round((track_timer-start),2)}s")

# %%
