"""Module provided by Luca Baroni."""

#%%
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as f
import csng_invariances._dnn_blocks as bl
import numpy as np


def gauss(x, y, mux, muy, A, sigma):
    """compute gaussian function"""
    x = x - mux
    y = y - muy
    return A * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


class DogLayer(nn.Module):
    def __init__(self, num_units, input_size_x, input_size_y):
        """Difference of gaussian (dog) layer
        based on NDN3 Peter Houska implementation: https://github.com/NeuroTheoryUMD/NDN3/blob/master/layer.py
        and HSM model: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927
        Args:
            input_size_x (int): number of pixels in the x axis
            input_size_y (int): number of pixels in y axis
            num_units (int): number of difference of gaussians units
        """
        super().__init__()
        self.num_units = num_units
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        x_arr = torch.linspace(-1, 1, self.input_size_x)
        y_arr = torch.linspace(-1, 1, self.input_size_y)
        self.X, self.Y = torch.meshgrid(x_arr, y_arr, indexing="ij")
        self.X = self.X.unsqueeze(0)
        self.Y = self.Y.unsqueeze(0)
        self.bounds = {
            "A": (torch.finfo(float).eps, None),
            "sigma": (
                1 / max(int(0.82 * self.input_size_x), int(0.82 * self.input_size_y)),
                1,
            ),
            "mu": (-0.8, 0.8),
        }

        # initialize
        self.mu_x = nn.Parameter(
            torch.zeros(self.num_units, 1, 1).uniform_(*self.bounds["mu"])
        )
        self.mu_y = nn.Parameter(
            torch.zeros(self.num_units, 1, 1).uniform_(*self.bounds["mu"])
        )
        self.A_c = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(self.bounds["A"][0], 10)
        )
        self.A_s = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(self.bounds["A"][0], 10)
        )
        self.sigma_1 = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(*self.bounds["sigma"])
        )
        self.sigma_2 = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(*self.bounds["sigma"])
        )

    def compute_dog(self):
        """clamp parameters to allowed values and generate dog filter

        Returns:
            torch.Tensor : dog filter size=[n_units, input_size_x, input_size_y]
        """
        self.clamp_parameters()
        sigma_c = self.sigma_1
        sigma_s = self.sigma_1 + self.sigma_2
        g_c = gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_c, sigma_c)
        g_s = gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_s, sigma_s)
        dog = g_c - g_s
        return dog

    def clamp_parameters(self):
        """clamp parameters to allowed bound values"""
        torch.clamp(self.mu_x, *self.bounds["mu"])
        torch.clamp(self.mu_y, *self.bounds["mu"])
        torch.clamp(self.A_c, *self.bounds["A"])
        torch.clamp(self.A_s, *self.bounds["A"])
        torch.clamp(self.sigma_1, *self.bounds["sigma"])
        torch.clamp(self.sigma_2, *self.bounds["sigma"])

    def get_dog_parameters(self, i=0):
        """get dictionary of dog parameters for given unit

        Args:
            i (int, optional): unit. Defaults to 0.

        Returns:
            dict: dictionary containing parameters
        """
        params = {
            "A_c": self.A_c[i].squeeze().detach().cpu().numpy(),
            "A_s": self.A_s[i].squeeze().detach().cpu().numpy(),
            "sigma_c": self.sigma_1[i].squeeze().detach().cpu().numpy(),
            "sigma_s": (self.sigma_1[i] + self.sigma_2[i])
            .squeeze()
            .detach()
            .cpu()
            .numpy(),
        }
        return params

    def plot_dog(self, n=None):
        """plot first n dog unit filters

        Args:
            n (int, optional): number of filters to plot. Defaults to 1.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if n == None:
            n = self.num_units
        for i in range(n):
            dog = self.compute_dog()[i].detach().cpu().numpy()
            max_abs_dog = np.max(np.abs(dog))
            p = self.get_dog_parameters(i)
            plt.imshow(dog, cmap=cm.coolwarm)
            plt.title(
                f"A_c = {p['A_c']:.2f},\n"
                + f"A_s = {p['A_s']:.2f},\n"
                + f"sigma_c = {p['sigma_c']:.2f},\n"
                + f"sigma_s = {p['sigma_s']:.2f},\n"
                + f"max = {np.max(dog):.2f},\n"
                + f"min = {np.min(dog):.2f}",
                loc="left",
            )
            plt.clim(-max_abs_dog, max_abs_dog)
            plt.colorbar()
            plt.show()

    def forward(self, x):
        dogs = self.compute_dog()
        x = torch.tensordot(x, dogs, dims=((1, 2), (1, 2)))
        return x


class HSMActivation(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: str = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        thresholds = torch.empty(
            size=(self.input_size, self.output_size),
            device=self.device,
            dtype=torch.float,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is vector
        x = torch.log10(1 + torch.exp(x - self.ti_s))


#%%
#
