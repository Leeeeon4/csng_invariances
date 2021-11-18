# %%
import torch
import torch.nn as nn
from rich.progress import track
import numpy as np


class Mask(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_size, self.channels, self.height, self.width = x.shape


def compute_mask(image, response, encoding_model):
    activation = encoding_model(image)
    num_different_pixels = 20
    stds = torch.empty(
        size=(response.shape[0], image.shape[2], image.shape[3]), device="cuda"
    )
    with torch.no_grad():
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
                    activation_differences[:, counter] = (output - activation).squeeze()
                    del img_copy, output
                stds[:, i, j] = torch.std(activation_differences, dim=1)
                del activation_differences
    return stds
