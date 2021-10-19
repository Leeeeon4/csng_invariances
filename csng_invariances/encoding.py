"""Module provding CNN encoding functionality."""

import wandb
from datasets.lurz2020 import download_lurz2020_data, static_loaders
from models.discriminator import get_core_trained_model
from training.trainers import standard_trainer as trainer
import torch
import argparse
from rich import print
from pathlib import Path
from datetime import datetime


def encode():
    """Wrap training.

    Returns:
        tuple: tuple of model, dataloaders, device, dataset_config
    """

    def sweep_parser():
        """Handle argparsing

        Returns:
            namespace: Namespace of parsed arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--seed", type=int, default=1, help="Seed for randomness. Defaults to 1."
        )
        parser.add_argument(
            "--avg_loss",
            type=bool,
            default=False,
            help=(
                "If True, the loss is averaged/summed over one batch. Defaults to False."
            ),
        )
        parser.add_argument(
            "--scale_loss",
            type=bool,
            default=True,
            help=(
                "If True, the loss is scaled according to the dataset size. "
                "Defaults to True."
            ),
        )
        parser.add_argument(
            "--loss_function",
            type=str,
            default="PoissonLoss",
            help=("Loss function to use. Defaults to PossionLoss."),
        )
        parser.add_argument(
            "--loss_accum_batch_n",
            type=int,
            default=None,
            help="Number of batches to accumulate the loss over. Defaults to None.",
        )
        parser.add_argument(
            "--stop_function",
            type=str,
            default="get_correlations",
            help=(
                "the function (metric) that is used to determine the end of the "
                "training in early stopping. Defaults to get_correlation."
            ),
        )
        parser.add_argument(
            "--device",
            type=str,
            help="Device to run training on. Defaults to cuda if possible.",
        )
        parser.add_argument(
            "--verbose",
            type=bool,
            default=True,
            help=(
                "If True, prints out a message for each optimzer step. Defaults to True."
            ),
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
            "--epoch", type=int, default=0, help=("Starting epoch. Defaults to 0.")
        )
        parser.add_argument(
            "--lr_init",
            type=float,
            default=0.005,
            help=("Initial learning rate. Defaults to 0.005."),
        )
        parser.add_argument(
            "--max_iter",
            type=int,
            default=200,
            help=("Maximum number of training iterations. Defaults to 200."),
        )
        parser.add_argument(
            "--maximize",
            type=bool,
            default=True,
            help=("If True, maximize the objective function. Defaults to True."),
        )
        parser.add_argument(
            "--tolerance",
            type=float,
            default=1e-6,
            help=("Tolerance for early stopping. Defaults to 1e-6."),
        )
        parser.add_argument(
            "--restore_best",
            type=bool,
            default=True,
            help=(
                "Whether to restore the model to the best state after early "
                "stopping. Defaults to True."
            ),
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
            "--cb",
            type=bool,
            default=None,
            help=("whether to execute callback function. Defaults to None."),
        )
        parser.add_argument(
            "--track_training",
            type=bool,
            default=True,
            help=(
                "whether to track and print out the training progress. Defaults to True."
            ),
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
        args = parser.parse_args()
        return args

    def train_encoding(
        seed,
        avg_loss=False,
        scale_loss=True,
        loss_function="PoissonLoss",
        stop_function="get_correlations",
        loss_accum_batch_n=None,
        device="cuda",
        verbose=True,
        interval=1,
        patience=5,
        epoch=0,
        lr_init=0.005,
        max_iter=200,
        maximize=True,
        tolerance=1e-6,
        restore_best=True,
        lr_decay_steps=3,
        lr_decay_factor=0.3,
        min_lr=0.0001,
        cb=None,
        track_training=False,
        return_test_score=False,
        detach_core=False,
        batch_size=64,
        **kwargs,
    ):
        """Train the encoding model.

        Args:
            args (namespace): Namespace of parsed arguments.

        Returns:
            tuple: tuple of model, dataloaders, device, dataset_config
        """
        trainer_config = locals()
        del trainer_config["kwargs"]
        # Handle device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda = False if str(device) == "cpu" else True
        print(f"Running the model on {device} with cuda: {cuda}")

        # Load data and model
        # TODO Include batchsize in wandb for sweeps
        lurz_dir = Path.cwd() / "data" / "external" / "lurz2020"
        if (lurz_dir / "README.md").is_file() is False:
            download_lurz2020_data()
        # Building Dataloaders
        dataset_config = {
            "paths": [str(lurz_dir / "static20457-5-9-preproc0")],
            "batch_size": batch_size,
            "seed": seed,
            "cuda": cuda,
            "normalize": True,
            "exclude": "images",
        }
        dataloaders = static_loaders(**dataset_config)
        model = get_core_trained_model(dataloaders)

        # If you want to allow fine tuning of the core, set detach_core to False
        if detach_core:
            print("Core is fixed and will not be fine-tuned")
        else:
            print("Core will be fine-tuned")

        # Display traininer config

        trainer_config["device"] = device
        print("Running current training config:")
        print(f"{dict(trainer_config)}")

        # Run trainer
        score, output, model_state = trainer(
            model=model, dataloaders=dataloaders, **trainer_config
        )

        return model, dataloaders, device, dataset_config

    wandb.init(project="invariances_encoding_LurzModel", entity="csng-cuni")
    config = wandb.config
    kwargs = sweep_parser()
    config.update(kwargs)
    model, dataloaders, device, dataset_config = train_encoding(**vars(kwargs))
    return model, dataloaders, device, dataset_config


def evaluate_encoding(model, dataloaders, device, dataset_config):
    """Evalutes the trained encoding model.

    Args:
        model (Encoder): torch.nn.Module inherited class Encoder.
        dataloaders (OrderedDict): dict of train, validation and test
            DataLoader objects
        device (str): String of device to use for computation
        dataset_config (dict): dict of dataset options
    """
    # Performane
    from utility.measures import get_correlations, get_fraction_oracles

    train_correlation = get_correlations(
        model, dataloaders["train"], device=device, as_dict=False, per_neuron=False
    )
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
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


if __name__ == "__main__":
    model, dataloaders, device, dataset_config = encode()
    t = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    model_path = Path.cwd() / "models" / t
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path / "trained_model.pth")
    # evaluate_encoding(model, dataloaders, device, dataset_config)
