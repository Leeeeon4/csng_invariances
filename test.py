# %%
import torch
import wandb

from pathlib import Path
from rich import print
import wandb

from csng_invariances.datasets.lurz2020 import download_lurz2020_data, static_loaders
from csng_invariances.models.discriminator import (
    download_pretrained_lurz_model,
    se2d_fullgaussian2d,
)
from csng_invariances.training.trainers import standard_trainer as lurz_trainer


# %%
# to be done by argparsing
batch_size = 64
seed = 1
interval = 1
patience = 5
lr_init = 0.005
max_iter = 200
tolerance = 1e-6
lr_decay_steps = 3
lr_decay_factor = 0.3
min_lr = 0.0001
detach_core = True
# %%
# read from system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = False if str(device) == "cpu" else True

# %%
# Settings
lurz_data_path = Path.cwd() / "data" / "external" / "lurz2020"
lurz_model_path = Path.cwd() / "models" / "external" / "lurz2020"
dataset_config = {
    "paths": [str(lurz_data_path / "static20457-5-9-preproc0")],
    "batch_size": batch_size,
    "seed": seed,
    "cuda": cuda,
    "normalize": True,
    "exclude": "images",
}
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
trainer_config = {
    "avg_loss": False,
    "scale_loss": True,
    "loss_function": "PoissonLoss",
    "stop_function": "get_correlations",
    "loss_accum_batch_n": None,
    "verbose": True,
    "maximize": True,
    "restore_best": True,
    "cb": None,
    "track_training": True,
    "return_test_score": False,
    "epoch": 0,
    "device": device,
    "seed": seed,
    "detach_core": detach_core,
    "batch_size": batch_size,
    "lr_init": lr_init,
    "lr_decay_factor": lr_decay_factor,
    "lr_decay_steps": lr_decay_steps,
    "min_lr": min_lr,
    "max_iter": max_iter,
    "tolerance": tolerance,
    "interval": interval,
    "patience": patience,
}
# %%
# Load data and model
download_lurz2020_data() if (lurz_data_path / "README.md").is_file() is False else None
download_pretrained_lurz_model() if (
    lurz_model_path / "transfer_model.pth.tar"
).is_file() is False else None

# %%
print(f"Running current dataset config:\n{dataset_config}")
dataloaders = static_loaders(**dataset_config)
# %%
print(f"Running current model config:\n{model_config}")
model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
transfer_model = torch.load(
    Path.cwd() / "models" / "external" / "lurz2020" / "transfer_model.pth.tar",
    map_location=torch.device("cpu"),
)
model.load_state_dict(transfer_model, strict=False)

# %%
print(f"Running current training config:\n{trainer_config}")
wandb.init()
config = wandb.config
kwargs = dict(dataset_config, **model_config)
kwargs.update(trainer_config)
config.update(kwargs)
print(kwargs)
score, output, model_state = lurz_trainer(
    model=model, dataloaders=dataloaders, **trainer_config
)

# %%
