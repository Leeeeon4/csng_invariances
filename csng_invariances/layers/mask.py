"""Submodule providing different masking layers."""

import torch
from rich.progress import track
from typing import Tuple
from numpy import linspace, load, save
from pathlib import Path
from csng_invariances._utils.utlis import string_time


class NaiveMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, neuron: int, device: str = None) -> None:
        """Instatiate masking module.

        Module for masking a tensor.

        Args:
            mask (torch.Tensor): Binary mask of shape (num_neuron, height, width)
            neuron (int): Neuron to use.
            device (str, optional): torch device, if None tries cuda. Defaults to None.
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.mask = mask[neuron, :, :].reshape(1, 1, mask.shape[1], mask.shape[2]).int()
        self.mask = self.mask.to(self.device)
        print(self.mask.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Masked tensor.
        """
        return x * self.mask

    @staticmethod
    def compute_mask(
        one_image: torch.Tensor,
        one_response: torch.Tensor,
        encoding_model: torch.nn.Module,
        device: str = None,
        num_different_pixels: int = 20,
        threshold: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a binary mask of pixels not influencing neural activation.

        Present stimuli to the encoding model and compute activations.
        After that, for each pixel, we carry out the following steps:
            - For every one_image we produce its copy with a pixel changed to a specific
            value from some test range.
            - Every processed one_image is then presented to the encoding model.
            - For each pixel, we compute activation for different pixel values across
            the test range. By subtracting the original activation, we can measure
            the standard deviation of these differences.
        Pixels, where a low standard deviation of activation was measured, have a
        very low impact on the predicted activation. If these pixels were important,
        changing them would also cause a change in predicted activation. So we would
        get different values of activations and thus we would measure a higher standard
        deviation. Pixels with low standard deviation can therefore be masked out from
        the one_image. This approach is naive in a way that it assumes that pixels influence
        activation independently.
        (compare Kovacs 2021)

        Args:
            one_image (torch.Tensor): one one_image tensor. Not the whole dataset.
            one_response (torch.Tensor): one one_response tensor. Not the whole dataset.
            encoding_model (torch.nn.Module): encoding model.
            device (str, optional): device. Defaults to none.
            num_different_pixels (int, optional): number of different pixel values
                to use for computation of standard deviation. Defaults to 20.
            threshold (float, optional): Binary threshold value for masking.
                Defaults to 0.02.

        Returns:
            Tuple: Tuple of binary mask tensor and and pixel standard deviation (roi)
                tensor. Both of shape (num_neurons, height, width).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pixel_standard_deviations = torch.empty(
            size=(one_response.shape[0], one_image.shape[2], one_image.shape[3]),
            device=device,
        )
        with torch.no_grad():
            activation = encoding_model(one_image)
            for i in track(range(one_image.shape[2])):
                for j in range(one_image.shape[3]):
                    activation_differences = torch.empty(
                        size=(one_response.shape[0], num_different_pixels)
                    )
                    for counter, value in enumerate(
                        linspace(0, 255, num_different_pixels)
                    ):
                        img_copy = one_image
                        img_copy[:, :, i, j] = value
                        output = encoding_model(img_copy)
                        activation_differences[:, counter] = (
                            output - activation
                        ).squeeze()
                        del img_copy, output
                    pixel_standard_deviations[:, i, j] = torch.std(
                        activation_differences, dim=1
                    )
                    del activation_differences
            mask = pixel_standard_deviations.ge(threshold)
        t = string_time()
        mask_directory = Path.cwd() / "models" / "masks" / t
        mask_directory.mkdir(parents=True, exist_ok=True)
        roi_directory = Path.cwd() / "reports" / "roi" / t
        roi_directory.mkdir(parents=True, exist_ok=True)
        save(mask_directory / "mask.npy", mask.detach().cpu().numpy())
        save(
            roi_directory / "pixel_standard_deviation.npy",
            pixel_standard_deviations.detach().cpu().numpy(),
        )
        with open(mask_directory / "readme.md", "w") as f:
            f.write(
                f"#Readme\n"
                f"The mask was based on the pixel standard deviation in: "
                f"{roi_directory}\n. The threshold used was {threshold}."
            )
        return mask, pixel_standard_deviations

    @staticmethod
    def load_binary_mask(path: str, device: str = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            data = load(path)
            data_tensor = torch.from_numpy(data)
            data_tensor.to(device)
        except Exception as err:
            print("An error occured. Is path correct? Is numpy file?\n\n" f"{err}")
        return data_tensor

    @staticmethod
    def load_pixel_standard_deviations(path: str, device: str = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            data = load(path)
            data_tensor = torch.from_numpy(data)
            data_tensor.to(device)
        except Exception as err:
            print("An error occured. Is path correct? Is numpy file?\n\n" f"{err}")
        return data_tensor
