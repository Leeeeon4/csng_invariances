"""Module provding CNN encoding functionality."""

from datasets.lurz2020 import get_dataloaders, static_loaders
from models.discriminator import get_core_trained_model
from training.trainers import standard_trainer as trainer
import torch


def train_encoding():
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
    detach_core = False
    if detach_core:
        print("Core is fixed and will not be fine-tuned")
    else:
        print("Core will be fine-tuned")

    trainer_config = {
        "track_training": True,
        "detach_core": detach_core,
        "max_iter": 200,
    }

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
    train_encoding()
