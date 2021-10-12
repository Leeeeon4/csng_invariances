"""Module provding CNN encoding functionality."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import neuralpredictors as neur
from datasets.lurz2020 import get_dataloaders
from models.discriminator import get_core_trained_model
from training.trainers import standard_trainer as trainer

def main():
    # Load data and model
    dataloaders = get_dataloaders()
    model = get_core_trained_model(dataloaders)

    # If you want to allow fine tuning of the core, set detach_core to False
    detach_core=True
    if detach_core:
        print('Core is fixed and will not be fine-tuned')
    else:
        print('Core will be fine-tuned')

    trainer_config = {'track_training': True,
                    'detach_core': detach_core}

    # Run trainer
    score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=1, **trainer_config)


if __name__ == "__main__":
    main()
