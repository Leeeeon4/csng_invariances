# %%
import torch
import torchvision
import numpy as np

from csng_invariances.datasets.lurz2020 import download_lurz2020_data, static_loaders
from csng_invariances.models.discriminator import (
    download_pretrained_lurz_model,
    se2d_fullgaussian2d,
)
from csng_invariances.training.trainers import standard_trainer as lurz_trainer
from csng_invariances.utility.ipyhandler import automatic_cwd

# %%
batch_size = 64
seed = 1
cuda = True
detach_core = True
# %%
# Load data and model
lurz_data_path = automatic_cwd() / "data" / "external" / "lurz2020"
lurz_model_path = automatic_cwd() / "model" / "external" / "lurz2020"

if (lurz_data_path / "README.md").is_file() is False:
    download_lurz2020_data()
if (lurz_model_path / "transfer_model.pth.tar").is_file() is False:
    download_pretrained_lurz_model()

# %%
dataset_config = {
    "paths": [str(lurz_data_path / "static20457-5-9-preproc0")],
    "batch_size": batch_size,
    "seed": seed,
    "cuda": cuda,
    "normalize": True,
    "exclude": "images",
}
dataloaders = static_loaders(**dataset_config)

model_config = {
    "init_mu_range": 0.55,
    "init_sigma": 0.4,
    "input_kern": 15,
    "hidden_kern": 13,
    "gamma_input": 1.0,
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 0,
        "final_tanh": False,
    },
    "gamma_readout": 2.439,
}

model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=1)
# Download pretrained model if not there
if (
    automatic_cwd() / "models" / "external" / "lurz2020" / "transfer_model.pth.tar"
).is_file() is False:
    download_pretrained_lurz_model()
# load model
transfer_model = torch.load(
    automatic_cwd() / "models" / "external" / "lurz2020" / "transfer_model.pth.tar",
    map_location=torch.device("cpu"),
)
model.load_state_dict(transfer_model, strict=False)


# If you want to allow fine tuning of the core, set detach_core to False
if detach_core:
    print("Core is fixed and will not be fine-tuned")
else:
    print("Core will be fine-tuned")
# %%
