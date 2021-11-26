# %%

import torch
import torch.nn as nn
from rich.progress import track
from typing import Tuple
import numpy as np
from pathlib import Path
from csng_invariances._utils.utlis import string_time


class NaiveMask(nn.Module):
    def __init__(self, mask: torch.Tensor, neuron: int) -> None:
        """Instatiate masking module.

        Module for masking a tensor.

        Args:
            mask (torch.Tensor): Binary mask of shape (num_neuron, height, width)
            neuron (int): Neuron to use.
        """
        super().__init__()
        self.mask = mask[neuron, :, :].reshape(1, 1, mask.shape[1], mask.shape[2]).int()

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
        image: torch.Tensor,
        response: torch.Tensor,
        encoding_model: torch.nn.Module,
        device: str = None,
        num_different_pixels: int = 20,
        threshold: float = 0.02,
    ) -> Tuple:
        """Compute a binary mask of pixels not influencing neural activation.

        Present stimuli to the encoding model and compute activations.
        After that, for each pixel, we carry out the following steps:
            - For every image we produce its copy with a pixel changed to a specific
            value from some test range.
            - Every processed image is then presented to the encoding model.
            - For each pixel, we compute activation for different pixel values across
            the test range. By subtracting the original activation, we can measure
            the standard deviation of these differences.
        Pixels, where a low standard deviation of activation was measured, have a
        very low impact on the predicted activation. If these pixels were important,
        changing them would also cause a change in predicted activation. So we would
        get different values of activations and thus we would measure a higher standard
        deviation. Pixels with low standard deviation can therefore be masked out from
        the image. This approach is naive in a way that it assumes that pixels influence
        activation independently.
        (compare Kovacs 2021)

        Args:
            image (torch.Tensor): image tensor.
            response (torch.Tensor): response tensor.
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
            size=(response.shape[0], image.shape[2], image.shape[3]), device=device
        )
        with torch.no_grad():
            activation = encoding_model(image)
            for i in track(range(image.shape[2])):
                for j in range(image.shape[3]):
                    activation_differences = torch.empty(
                        size=(response.shape[0], num_different_pixels)
                    )
                    for counter, value in enumerate(
                        np.linspace(0, 255, num_different_pixels)
                    ):
                        img_copy = image
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
        roi_directory = Path.cwd() / "models" / "roi" / t
        roi_directory.mkdir(parents=True, exist_ok=True)
        np.save(mask_directory / "mask.npy", mask.detach().cpu().numpy())
        np.save(
            roi_directory / "pixel_standard_deviation.npy",
            pixel_standard_deviations.detach().cpu().numpy(),
        )
        with open(mask_directory / "readme.md") as f:
            f.write(
                f"#Readme\n"
                f"The mask was based on the pixel standard deviation in: "
                f"{roi_directory}\n. The threshold used was {threshold}."
            )
        return mask, pixel_standard_deviations

    # TODO @staticmethod load binary NaiveMask
    # TODO @staticmethod load pixel_standard_deviations
