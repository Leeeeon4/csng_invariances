"""Module providing training loops for the generator model."""

import torch
import wandb
import matplotlib.pyplot as plt
from numpy import save

from rich import print
from rich.progress import track
from pathlib import Path
from typing import Callable, List, Tuple
from csng_invariances._utils.utlis import string_time
from csng_invariances._utils.video_maker import make_video
from csng_invariances.data._data_helpers import save_configs, scale_tensor_to_0_1

from csng_invariances.models.gan import GANModel
from csng_invariances.layers.loss_function import SelectedNeuronActivation
from csng_invariances.layers.mask import NaiveMask


class Trainer:
    def __init__(
        self,
        generator_model: torch.nn.Module,
        encoding_model: torch.nn.Module,
        data: list,
        mask: torch.Tensor = None,
        image_preprocessing: Callable = None,
        response_preprocessing: Callable = None,
        device: str = None,
        epochs: int = 50,
        *arg: int,
        **kwargs: dict,
    ) -> None:
        """
        Args:
            generator_model (torch.nn.Module): generator model.
            encoding_model (torch.nn.Module): encoding model
            data (list): list of latent_vectors.
                latent vectors are of shape (latent_dimension, batch_size).
                data is of len(batches).
            mask (torch.nn.Module, optional): masking layer. Defaults to None.
            image_preprocessing (Callable, optional): image preprocessing function.
                Defaults to None.
            response_preprocessing (Callable, optional): response preprocessing
                function. Defaults to None.
            device (str, optional): torch device, if None tries cuda. Defaults to
                None.
            epochs (int, optional): number of epochs. Defaults to 50.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.data = data
        self.latent_space_dimension = self.data[0].shape[1]
        self.epochs = epochs
        self.batches = len(self.data)
        self.batch_size = self.data[0].shape[0]
        self.generator_model = generator_model
        self.encoding_model = encoding_model
        if mask is None:
            mask = torch.ones(
                size=(5335, 36, 64),
                dtype=torch.float,
                requires_grad=True,
                device=self.device,
            )
        self.mask = mask
        for obj in [self.mask, self.encoding_model, self.generator_model]:
            obj = obj.to(self.device)
        self.image_preprocessing = image_preprocessing
        self.response_preprocessing = response_preprocessing
        self.config = kwargs


class NaiveTrainer(Trainer):
    def __init__(
        self,
        generator_model: torch.nn.Module,
        encoding_model: torch.nn.Module,
        data: List[torch.Tensor],
        mask: torch.Tensor = None,
        image_preprocessing: Callable = None,
        response_preprocessing: Callable = None,
        epochs: int = 50,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.001,
        show_development: bool = False,
        prep_video: bool = False,
        *arg: int,
        **kwargs: dict,
    ) -> None:
        """
        Args:
            generator_model (torch.nn.Module): generator model.
            encoding_model (torch.nn.Module): encoding model
            data (list): list of latent_vectors.
                latent vectors are of shape (latent_dimension, batch_size).
                data is of len(batches).
            mask (torch.nn.Module, optional): masking layer. Defaults to None.
            image_preprocessing (Callable, optional): image preprocessing function.
                Defaults to None.
            response_preprocessing (Callable, optional): response preprocessing
                function. Defaults to None.
            epochs (int, optional): number of epochs. Defaults to 50.
            lr (float, optional): learning rate. Defaults to 0.0001.
            betas (Tuple[float, float], optional): beta1 and beta2. Defaults to (0.9, 0.999).
            eps (float, optional): epsilon. Defaults to 1e-8.
            weight_decay (float, optional): weight decay - a.k.a. L2-norm reg. factor.
                Defaults to 0.001.
            show_development (bool, optional): if True, samples during training are
                stored as *.npy and *.png
            prep_video (bool, optional): if True, stores images during training in
                a way which allows for easy video creation.
        """
        super().__init__(
            generator_model,
            encoding_model,
            data,
            mask=mask,
            image_preprocessing=image_preprocessing,
            response_preprocessing=response_preprocessing,
            epochs=epochs,
            *arg,
            **kwargs,
        )
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay  # a weight decay of 0.001 and 0.01 lead to highly overfitted generator for latent dim 128, batchsize = 64 and num_batches = 16
        self.show_development = show_development
        self.prep_video = prep_video
        if self.prep_video:
            self.show_development = True

    def train(self, selected_neuron: int):
        run = wandb.init(
            project="invariances_generator_growing_linear_generator", entity="leeeeon4"
        )
        t = string_time()
        config = wandb.config
        self.config["selected_neuron"] = selected_neuron
        config.update(self.config, allow_val_change=True)
        self.masking_layer = NaiveMask(self.mask, selected_neuron)
        self.masking_layer = self.masking_layer.to(self.device)
        gan = GANModel(
            self.generator_model,
            self.encoding_model,
            self.masking_layer,
            self.image_preprocessing,
            self.response_preprocessing,
        )
        gan = gan.to(self.device)
        optimizer = torch.optim.Adam(
            params=self.generator_model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        loss_function = SelectedNeuronActivation()
        loss_function = loss_function.to(self.device)
        running_loss = 0.0
        print("Epoch 0:")
        for epoch in range(self.epochs):
            for batch in track(
                range(self.batches),
                total=self.batches,
                description=f"Epoch {epoch+1}/{self.epochs}:",
            ):
                optimizer.zero_grad()
                inputs = self.data[batch]

                inputs = inputs.to(self.device)
                activations, preprocessed_sample, masked_sample, sample = gan(inputs)
                activations = activations.to(self.device)
                loss = loss_function(activations, selected_neuron)
                loss.backward()
                wandb.log({"loss": loss})
                optimizer.step()
                running_loss += loss.item()
            print(
                "============================================\n"
                f"Epoch {epoch +1}, "
                f"average neural activation: "
                f"{round(abs(running_loss/(self.batches*self.batch_size)),5)}\n"
                "============================================"
            )
            if self.show_development:
                image_directory = (
                    Path.cwd()
                    / "reports"
                    / "figures"
                    / "generator"
                    / "during_training"
                    / t
                )
                image_directory.mkdir(parents=True, exist_ok=True)
                data_directory = (
                    Path.cwd()
                    / "data"
                    / "processed"
                    / "generator"
                    / "during_training"
                    / t
                    / f"neuron_{selected_neuron}"
                )
                data_directory.mkdir(parents=True, exist_ok=True)
                save(
                    file=data_directory / f"Epoch_{epoch}_00_sample.npy",
                    arr=sample.detach().cpu().numpy(),
                )
                plt.imshow(sample[0, :, :, :].detach().cpu().squeeze())
                plt.title("First sample in batch")
                plt.colorbar()
                plt.savefig(image_directory / f"Epoch_{epoch}_00_sample.jpg")
                plt.close()
                save(
                    file=data_directory / f"Epoch_{epoch}_01_masked_sample.npy",
                    arr=masked_sample.detach().cpu().numpy(),
                )
                plt.imshow(masked_sample[0, :, :, :].detach().cpu().squeeze())
                plt.title("First masked sample in batch")
                plt.colorbar()
                plt.savefig(image_directory / f"Epoch_{epoch}_01_masked_sample.jpg")
                plt.close()
                save(
                    file=data_directory / f"Epoch_{epoch}_02_preprocessed_sample.npy",
                    arr=preprocessed_sample.detach().cpu().numpy(),
                )
                # scale images for displaying
                pre_img = (
                    scale_tensor_to_0_1(preprocessed_sample[0, :, :, :])
                    .detach()
                    .cpu()
                    .squeeze()
                )
                plt.imshow(pre_img)
                plt.title(f"Epoch: {(epoch+1):02d}")
                plt.savefig(
                    image_directory / f"Epoch_{epoch}_02_preprocessed_sample.jpg"
                )
                if self.prep_video:
                    video_directory = Path.cwd() / "video" / f"neuron_{selected_neuron}"
                    video_directory.mkdir(parents=True, exist_ok=True)
                    plt.savefig(video_directory / f"{epoch:02d}.jpg")
                plt.close()
                if (epoch + 1) == self.epochs:
                    print(f"Images during training are stored in: {image_directory}")
                    print(f"Data during training is stored in: {data_directory}")
                    if self.prep_video:
                        make_video(
                            string_time=t,
                            generator_name=type(self.generator_model).__name__,
                            batch_size=self.batch_size,
                            num_batches=self.batches,
                            latent_space_dimension=self.latent_space_dimension,
                            neuron=selected_neuron,
                            epochs=self.epochs,
                            lr=self.lr,
                            weight_decay=self.weight_decay,
                        )
                        print(
                            f"Images for video creation are stored in: {video_directory}"
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
