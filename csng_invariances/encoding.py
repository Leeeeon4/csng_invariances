"""Module provding CNN encoding functionality."""

from torch._C import BoolType
from datasets.lurz2020 import get_dataloaders, static_loaders
from models.discriminator import get_core_trained_model
from training.trainers import standard_trainer as trainer
import torch
import argparse
from rich import print


def train_encoding(
    detach_core=True,
    avg_loss=False,
    scale_loss=False,
    loss_accum_batch_n=None,
    interval=1,
    patience=5,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
):
    """Train the encoding model.

    Returns:
        tuple: tuple of model, dataloaders, device, dataset_config
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running the model on {device}")
    if device == "cpu":
        cuda = False
    else:
        cuda = True
    # Load data and model
    dataloaders, dataset_config = get_dataloaders(cuda)
    model = get_core_trained_model(dataloaders)

    # If you want to allow fine tuning of the core, set detach_core to False
    if detach_core:
        print("Core is fixed and will not be fine-tuned")
    else:
        print("Core will be fine-tuned")

    # trainer_config = {"track_training": True, "detach_core": detach_core}

    trainer_config = {
        "avg_loss": avg_loss,
        "scale_loss": scale_loss,
        "loss_function": "PoissonLoss",
        "stop_function": "get_correlations",
        "loss_accum_batch_n": loss_accum_batch_n,
        "device": device,
        "verbose": True,
        "interval": interval,
        "patience": patience,
        "epoch": 0,
        "lr_init": lr_init,
        "max_iter": max_iter,
        "maximize": maximize,
        "tolerance": tolerance,
        "restore_best": True,
        "lr_decay_steps": lr_decay_steps,
        "lr_decay_factor": lr_decay_factor,
        "min_lr": min_lr,
        "cb": None,
        "track_training": True,
        "return_test_score": False,
        "detach_core": detach_core,
    }
    print(f"Running current training config:\n{trainer_config}")

    # Run trainer
    score, output, model_state = trainer(
        model=model, dataloaders=dataloaders, seed=1, **trainer_config
    )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg_loss", type=bool)
    parser.add_argument("--scale_loss", type=bool)
    parser.add_argument("--detach_core", type=bool)
    parser.add_argument("--loss_accum_batch_n", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr_init", type=float)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--maximize", type=bool)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--lr_decay_steps", type=int)
    parser.add_argument("--lr_decay_factor", type=float)
    parser.add_argument("--min_lr", type=float)
    args = parser.parse_args()
    train_encoding(
        # args.detach_core,
        # args.avg_loss,
        # args.scale_loss,
        # args.loss_accum_batch_n,
        # args.interval,
        # args.patience,
        # args.lr_init,
        # args.max_iter,
        # args.maximize,
        # args.tolerance,
        # args.lr_decay_steps,
        # args.lr_decay_factor,
        # args.min_lr,
    )
