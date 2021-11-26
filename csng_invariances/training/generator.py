"""Generator training module."""

import torch
import torch.optim as optim
import wandb

from rich import print
from rich.progress import track
from pathlib import Path
from csng_invariances._utils.utlis import string_time

from csng_invariances.data._data_helpers import save_configs
from csng_invariances.models.gan import ComputeModel
from csng_invariances.layers.loss_function import SelectedNeuronActivation


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
        run = wandb.init(
            project="invariances_generator_growing_linear_generator", entity="leeeeon4"
        )
        config = wandb.config
        self.config["selected_neuron"] = selected_neuron
        config.update(self.config, allow_val_change=True)
        gan = ComputeModel(self.generator_model, self.encoding_model)
        optimizer = optim.Adam(self.generator_model.parameters())
        loss_function = SelectedNeuronActivation()
        running_loss = 0.0
        print("Epoch 0:")
        for epoch in range(self.epochs):
            for batch in track(
                range(self.batches),
                total=self.batches,
                description=f"Epoch {epoch}/{self.epochs}:",
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
                f"{round(abs(running_loss/(self.batches*self.batch_size)),5)}"
            )
            running_loss = 0.0
        t = string_time()
        generator_model_directory = Path.cwd() / "models" / "generator" / t
        generator_model_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.generator_model.state_dict(),
            generator_model_directory
            / f"Trained_generator_neuron_{selected_neuron}.pth",
        )
        save_configs(self.config, generator_model_directory)
        generator_report_directory = Path.cwd() / "reports" / "generator"
        generator_model_directory.mkdir(parents=True, exist_ok=True)
        encoding_report_path = generator_report_directory / "readme.md"
        if encoding_report_path.is_file() is False:
            generator_report_directory.mkdir(parents=True, exist_ok=True)
            with open(encoding_report_path, "w") as file:
                file.write(
                    "# Generator\n"
                    "Generator training was tracked using weights and biases. "
                    "Reports may be found at:\n"
                    "https://wandb.ai/csng-cuni/invariances_generator_growing_linear_generator\n"
                    "Reports are only accessible to members of the csng_cuni "
                    "group.\n"
                    "## Neuron specificity\n"
                    "Generators are neuron specific. Please filter for 'selected_neuron' "
                    "when looking at generator model performace, as all generators "
                    "are stored in one WandB project."
                )
        print(f"Model and configs are stored at {generator_model_directory}")
        run.finish()
        return self.generator_model
