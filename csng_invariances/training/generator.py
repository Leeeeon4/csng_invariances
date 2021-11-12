"""Generator models module."""

import torch
import torch.optim as optim
import wandb

from rich.progress import track
from csng_invariances.models.gan import ComputeModel
from csng_invariances.losses.generator import SelectedNeuronActivation


class Trainer:
    def __init__(
        self,
        generator_model: torch.nn.Module,
        encoding_model: torch.nn.Module,
        epochs: int,
        batches: int,
        batch_size: int,
        data: list,
        *arg: int,
        **kwargs: dict,
    ) -> None:
        self.data = data
        self.epochs = epochs
        self.batches = batches
        self.batch_size = batch_size
        self.generator_model = generator_model
        self.encoding_model = encoding_model
        self.config = kwargs


class NaiveTrainer(Trainer):
    def __init__(
        self,
        generator_model: torch.nn.Module,
        encoding_model: torch.nn.Module,
        epochs: int,
        batches: int,
        batch_size: int,
        data: list,
        *arg: int,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            generator_model,
            encoding_model,
            epochs,
            batches,
            batch_size,
            data,
            *arg,
            **kwargs,
        )

    def train(self, selected_neuron: int):
        wandb.init(
            project="invariances_generator_growing_linear_generator", entity="leeeeon4"
        )
        config = wandb.config
        self.config["selected_neuron"] = selected_neuron
        config.update(self.config)
        gan = ComputeModel(self.generator_model, self.encoding_model)
        optimizer = optim.Adam(self.generator_model.parameters())
        loss_function = SelectedNeuronActivation()
        running_loss = 0.0
        print("Epoch 0:")
        for epoch in range(self.epochs):
            for batch in track(
                range(self.batches),
                total=self.batches,
                description="Epoch {}".format(epoch),
            ):
                optimizer.zero_grad()
                activations = gan(self.data[batch])
                loss = loss_function(activations, selected_neuron)
                loss.backward()
                wandb.log({"loss": loss})
                optimizer.step()
                running_loss += loss.item()
            print(
                f"Epoch {epoch +1}:\n"
                f"Average neural activation: "
                f"{round(abs(running_loss/(self.batches*self.batch_size)),2)}"
            )
            running_loss = 0.0
        return self.generator_model
