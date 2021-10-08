"""Main module."""

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import neuralpredictors as neur
from pathlib import Path
from torch.utils.data.dataloader import DataLoader


def load_lurz_data():
    from datasets.lurz2020 import static_loaders

    dataloaders = static_loaders(
        **{
            "paths": [
                str(
                    Path.cwd().parents[0]
                    / "data"
                    / "external"
                    / "lurz2020"
                    / "static20457-5-9-preproc0"
                )
            ],
            "batch_size": 64,
            "seed": 1,
            "cuda": False,
            "normalize": True,
            "exclude": "images",
        }
    )
    return dataloaders


def load_antolik_data(region):
    import datasets.antolik2016 as al

    dataloaders = {}

    ds_al_train = al.Antolik2016Dataset(
        data_dir=Path.cwd() / "data" / "external" / "antolik2016" / "Data",
        region=region,
        dataset_type="training",
    )

    dataloaders["training"] = DataLoader(ds_al_train)

    ds_al_val = al.Antolik2016Dataset(
        data_dir=Path.cwd() / "data" / "external" / "antolik2016" / "Data",
        region=region,
        dataset_type="validation",
    )

    dataloaders["validation"] = DataLoader(ds_al_val)


def train_discriminator(dataloaders):
    from models.discriminator import se2d_fullgaussian2d

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

    transfer_model = torch.load(
        Path.cwd() / "models" / "external" / "transfer_model.pth.tar",
        map_location=torch.device("cpu"),
    )

    model.load_state_dict(transfer_model, strict=False)


def main():
    """Concept:
    1. Train discriminator of data
    2. Compute Predictions:
    - discriminator Predictions
    - linear Predictions
    3. Compute score
    4. Compute MEI
    5. Compute ROI
    6. Train generator n(0,1)
    7. Generate Samples u(-2,2)
    8. Cluster 'most representative/most different' samples
    9. Analyze important samples
    """

    # Setup Matplotlib style based on matplotlibrc-file.
    # If two plot are supposed to fit on one presentation slide, use figsize = figure_sizes["half"]
    mpl.rc_file("matplotlibrc")
    figure_sizes = {
        "full": (8, 5.6),
        "half": (5.4, 3.8),
    }


if __name__ == "__main__":
    main()
