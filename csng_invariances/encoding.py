"""Module provding CNN encoding functionality."""

import wandb
import torch
import torch.optim as optim
import argparse
import datetime
import json

from rich import print
from pathlib import Path
from torch import nn

from csng_invariances.datasets.lurz2020 import download_lurz2020_data, static_loaders
from csng_invariances.models.discriminator import (
    download_pretrained_lurz_model,
    se2d_fullgaussian2d,
)
from csng_invariances.training.trainers import standard_trainer as lurz_trainer
from csng_invariances.utility.data_helpers import save_configs, load_configs
from csng_invariances.utility.measures import get_correlations, get_fraction_oracles


def encoding_parser():
    """Handle argparsing of encoding sweeps.

    Returns:
        namespace: Namespace of parsed encoding arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lurz",
        help=(
            "Specify the dataset to analyze. Options are 'Lurz' and "
            "'Antolik'. Defaults to 'Lurz'."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Set device if automatic reading of device is not wanted. "
            "Options: 'cpu', 'cuda'. Defaults to reading from system."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=1, help=("Seed for randomness. Defaults to 1.")
    )
    parser.add_argument(
        "--detach_core",
        type=bool,
        default=True,
        help=("If True, the core will not be fine-tuned. Defaults to True."),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=("Size of batches. Defaults to 64."),
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.005,
        help=("Initial learning rate. Defaults to 0.005."),
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=3,
        help=(
            "How many times to decay the learning rate after no improvement. "
            "Defaults to 3."
        ),
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.3,
        help=("Factor to decay the learning rate with. Defaults to 0.3."),
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0001,
        help=("minimum learning rate. Defaults to 0.005."),
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help=("Maximum number of training iterations. Defaults to 200."),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help=(
            "Interval at which objective is evaluated to consider early "
            "stopping. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Number of times the objective is allowed to not become better "
            "before the iterator terminates. Defaults to 5."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help=("Tolerance for early stopping. Defaults to 1e-6."),
    )
    kwargs = parser.parse_args()
    return kwargs


def encode(parsed_kwargs):
    """Encode and evaluate model.

    Args:
        parsed_kwargs (namespace): Namespace of parsed config arguments.

    Returns:
        model: Trained encoding model.
    """

    def train_lurz_readout_encoding(
        seed,
        interval,
        patience,
        lr_init,
        tolerance,
        lr_decay_steps,
        lr_decay_factor,
        min_lr,
        batch_size,
        paths=[
            str(
                Path.cwd()
                / "data"
                / "external"
                / "lurz2020"
                / "static20457-5-9-preproc0"
            )
        ],
        normalize=True,
        exclude="images",
        init_mu_range=0.55,
        init_sigma=0.4,
        input_kern=15,
        hidden_kern=13,
        gamma_input=1.0,
        grid_mean_predictor={
            "type": "cortex",
            "input_dimensions": 2,
            "hidden_layers": 0,
            "hidden_features": 0,
            "final_tanh": False,
        },
        gamma_readout=2.439,
        avg_loss=False,
        scale_loss=True,
        loss_function="PoissonLoss",
        stop_function="get_correlations",
        loss_accum_batch_n=None,
        device="cpu",
        verbose=True,
        maximize=True,
        restore_best=True,
        cb=None,
        track_training=True,
        return_test_score=False,
        detach_core=False,
        epoch=0,
        max_iter=200,
        **kwargs,
    ):
        """Train the encoding model.

        The model is based on the Lurz et al. 2020 pretrained core. The readout
        is trained and saved.

        Args:
            seed (int): Seed for randomness.
            interval (int): interval at which objective is evaluated to
                consider early stopping.
            patience (int): number of times the objective is allowed to not
                become better before the iterator terminates.
            lr_init (float): initial learning rate.
            tolerance (float): tolerance for early stopping.
            lr_decay_steps (int): how many times to decay the learning
                rate after no improvement.
            lr_decay_factor (float): factor to decay the learning rate. Must be
                less than 1.
            min_lr (float): minimum learning rate.
            batch_size (int): batch size.
            paths (list, optional): list of lurz dataset paths. Defaults to
                [ str( Path.cwd() / "data" / "external" / "lurz2020" /
                "static20457-5-9-preproc0" ) ].
            normalize (bool, optional): whether to normalize the data (see also
                exclude). Defaults to True.
            exclude (str, optional): data to exclude from data-normalization.
                Only relevant if normalize=True. Defaults to "images".
            init_mu_range (float, optional): Lurz et al. 2020 readout parameter.
                Defaults to 0.55.
            init_sigma (float, optional): Lurz et al. 2020 readout parameter.
                Defaults to 0.4.
            input_kern (int, optional): Lurz et al. 2020 core parameter.
                Defaults to 15.
            hidden_kern (int, optional): Lurz et al. 2020 core parameter.
                Defaults to 13.
            gamma_input (float, optional): Lurz et al. 2020 core parameter.
                Defaults to 1.0.
            grid_mean_predictor: if not None, needs to be a dictionary of the form
                {
                'type': 'cortex',
                'input_dimensions': 2,
                'hidden_layers':0,
                'hidden_features':0,
                'final_tanh': False,
                }
                In that case the datasets need to have the property
                `neurons.cell_motor_coordinates`
            gamma_readout (float, optional): Lurz et al. 2020 readout parameter.
                Defaults to 2.439.
            avg_loss (bool, optional): whether to average (or sum) the loss over a
                batch. Defaults to False.
            scale_loss (bool, optional): hether to scale the loss according to the
                size of the dataset. Defaults to True.
            loss_function (str, optional): loss function to use. Defaults to
                'PoissonLoss'.
            stop_function (str, optional): the function (metric) that is used to
            determine the end of the training in early stopping. Defaults to
            'get_correlation'.
            loss_accum_batch_n (int, optional): number of batches to accumulate
                the loss over. Defaults to None.
            device (str, optional): Device to compute on. Defaults to "cpu".
            verbose (bool, optional): whether to print out a message for each
                optimizer step. Defaults to True.
            maximize (bool, optional): whether to maximize or minimize the
                objective function. Defaults to True.
            restore_best (bool, optional): whether to restore the model to the best
                state after early stopping. Defaults to True.
            cb ([type], optional): whether to execute callback function. Defaults to
                None.
            track_training (bool, optional): whether to track and print out the
                training progress. Defaults to True.
            return_test_score (bool, optional): Return the average validation
                correlation during evaluation. Defaults to False.
            detach_core (bool, optional): If true, the core is not trained.
                Defaults to False.
            epoch (int, optional): starting epoch. Defaults to 0.
            max_iter (int, optional): maximum number of training iterations.
                Defaults to 200.

        Returns:
            tuple: tuple of (model, dataloaders, configs), where configs is a
                dict of sub_config dicts.
        """
        assert lr_decay_factor <= 1, "lr_decay_factor must be less than 1."
        if device is None:
            # read from system
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cuda = False if str(device) == "cpu" else True
            print(f"Running the model on {device} with cuda: {cuda}")

        # settings
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

        # Download data and model if necessary
        download_lurz2020_data() if (
            lurz_data_directory / "README.md"
        ).is_file() is False else None
        download_pretrained_lurz_model() if (
            lurz_model_path
        ).is_file() is False else None

        # Load data
        print(
            f"Running current dataset config:\n{json.dumps(dataset_config, indent=2)}"
        )
        dataloaders = static_loaders(**dataset_config)

        # Model setup
        print(f"Running current model config:\n{json.dumps(model_config, indent=2)}")
        # build model
        model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=seed)
        # load state_dict of pretrained core
        transfer_model = torch.load(
            lurz_model_path,
            map_location=device,
        )
        model.load_state_dict(transfer_model, strict=False)

        # Training readout
        print_trainer_config = trainer_config
        print_trainer_config["device"] = str(print_trainer_config["device"])
        print(
            f"Running current training config:\n{json.dumps(print_trainer_config, indent=2)}"
        )
        wandb.init(project="invariances_encoding_LurzModel", entity="csng-cuni")
        config = wandb.config
        kwargs = dict(dataset_config, **model_config)
        kwargs.update(trainer_config)
        config.update(kwargs)
        lurz_trainer(model=model, dataloaders=dataloaders, **trainer_config)

        # Saving model (core + readout)
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        readout_model_directory = Path.cwd() / "models" / "encoding" / t
        readout_model_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            readout_model_directory / "Pretrained_core_readout_lurz.pth",
        )
        configs = {
            "dataset_config": dataset_config,
            "model_config": model_config,
            "trainer_config": trainer_config,
        }
        save_configs(configs, readout_model_directory)
        encoding_report_directory = Path.cwd() / "reports" / "encoding"
        encoding_report_path = encoding_report_directory / "readme.md"
        if encoding_report_path.is_file() is False:
            encoding_report_directory.mkdir(parents=True, exist_ok=True)
            with open(encoding_report_path, "w") as file:
                file.write(
                    "# Encoding\n"
                    "Encoding training was tracked using weights and biases. "
                    "Reports may be found at:\n"
                    "https://wandb.ai/csng-cuni/invariances_encoding_LurzModel\n"
                    "Reports are only accessible to members of the csng_cuni "
                    "group."
                )
        print(f"Model and configs are stored at {readout_model_directory}")
        return model, dataloaders, configs

    def evaluate_lurz_readout_encoding(model, dataloaders, configs):
        """Evalutes the trained encoding model.

        This functions is based on the sourcecode provided in the example
        notebook of Lurz et al. 2021: Generalization in Data-driven Models of
        Primary Visual Cortex.

        Args:
            model (Encoder): torch.nn.Module inherited class Encoder.
            dataloaders (OrderedDict): dict of train, validation and test
                DataLoader objects
            device (str): String of device to use for computation
            dataset_config (dict): dict of dataset options
        """
        dataset_config = configs["dataset_config"]
        device = configs["trainer_config"]["device"]
        # Performane

        train_correlation = get_correlations(
            model, dataloaders["train"], device=device, as_dict=False, per_neuron=False
        )
        validation_correlation = get_correlations(
            model,
            dataloaders["validation"],
            device=device,
            as_dict=False,
            per_neuron=False,
        )
        test_correlation = get_correlations(
            model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
        )

        # Fraction Oracle can only be computed on the test set. It requires the dataloader to give out batches of repeats of images.
        # This is achieved by building a dataloader with the argument "return_test_sampler=True"
        oracle_dataloader = static_loaders(
            **dataset_config, return_test_sampler=True, tier="test"
        )
        fraction_oracle = get_fraction_oracles(
            model=model, dataloaders=oracle_dataloader, device=device
        )[0]

        print("-----------------------------------------")
        print("Correlation (train set):      {0:.3f}".format(train_correlation))
        print("Correlation (validation set): {0:.3f}".format(validation_correlation))
        print("Correlation (test set):       {0:.3f}".format(test_correlation))
        print("-----------------------------------------")
        print("Fraction oracle (test set):   {0:.3f}".format(fraction_oracle))

    if vars(parsed_kwargs)["dataset"] == "Lurz":
        model, dataloaders, configs = train_lurz_readout_encoding(**vars(parsed_kwargs))
        evaluate_lurz_readout_encoding(model, dataloaders, configs)
    return model


def load_parser():
    """Handle encoding model directory parsing.

    Returns:
        str: Model directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoding_model_directory",
        type=str,
        help=(
            "Directory of the trained encoding model. Recall, the model must "
            "fit to the dataset, as the readout is dataset specific."
        ),
    )
    kwargs = parser.parse_args()
    return vars(kwargs)["encoding_model_directory"]


def load_encoding_model(model_directory):
    """Load pretrained encoding model.

    Args:
        model_directory (str): path to model directory to load from.

    Returns:
        model: Trained encoding model.
    """
    configs = load_configs(model_directory)
    dataloaders = static_loaders(**configs["dataset_config"])
    model = se2d_fullgaussian2d(
        **configs["model_config"],
        dataloaders=dataloaders,
        seed=configs["dataset_config"]["seed"],
    )
    model.load_state_dict(
        torch.load(Path(model_directory) / "Pretrained_core_readout_lurz.pth")
    )
    return model


if __name__ == "__main__":
    pass
