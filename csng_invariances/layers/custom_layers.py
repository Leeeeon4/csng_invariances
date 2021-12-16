"""Module provided by Luca Baroni."""

#%%
from typing import List
import torch
import torch.nn as nn


class CustomLayer(nn.Module):
    """Layer which automatically moves to cuda if available and not specified otherwise."""

    def __init__(self, device: str = None, *args, **kwargs) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


class HSMLayers(CustomLayer):
    """A parent class for a custom layers required for the HSM model as described in
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DifferenceOfGaussiangLayer(HSMLayers):
    def __init__(
        self, num_lgn_units: int, input_size_x: int, input_size_y: int, *args, **kwargs
    ) -> None:
        """Difference of gaussian (dog) layer
        based on NDN3 Peter Houska implementation: https://github.com/NeuroTheoryUMD/NDN3/blob/master/layer.py
        and HSM model: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927
        Args:
            input_size_x (int): number of pixels in the x axis
            input_size_y (int): number of pixels in y axis
            num_lgn_units (int): number of difference of gaussians units
        """
        super().__init__(*args, **kwargs)
        self.num_lgn_units = num_lgn_units
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        x_arr = torch.linspace(-1, 1, self.input_size_x, device=self.device)
        y_arr = torch.linspace(-1, 1, self.input_size_y, device=self.device)
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
            torch.zeros(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                *self.bounds["mu"]
            )
        )
        self.mu_y = nn.Parameter(
            torch.zeros(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                *self.bounds["mu"]
            )
        )
        self.A_c = nn.Parameter(
            torch.ones(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                self.bounds["A"][0], 10
            )
        )
        self.A_s = nn.Parameter(
            torch.ones(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                self.bounds["A"][0], 10
            )
        )
        self.sigma_1 = nn.Parameter(
            torch.ones(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                *self.bounds["sigma"]
            )
        )
        self.sigma_2 = nn.Parameter(
            torch.ones(self.num_lgn_units, 1, 1, device=self.device).uniform_(
                *self.bounds["sigma"]
            )
        )

    def gauss(self, x, y, mux, muy, A, sigma):
        """compute gaussian function"""
        x = x - mux
        y = y - muy
        return A * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    def compute_dog(self) -> torch.Tensor:
        """clamp parameters to allowed values and generate dog filter

        Returns:
            torch.Tensor : dog filter size=[num_lgn_units, input_size_x, input_size_y]
        """
        self.clamp_parameters()
        sigma_c = self.sigma_1
        sigma_s = self.sigma_1 + self.sigma_2
        g_c = self.gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_c, sigma_c)
        g_s = self.gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_s, sigma_s)
        dog = g_c - g_s
        return dog

    def clamp_parameters(self) -> None:
        """clamp parameters to allowed bound values"""
        torch.clamp(self.mu_x, *self.bounds["mu"])
        torch.clamp(self.mu_y, *self.bounds["mu"])
        torch.clamp(self.A_c, *self.bounds["A"])
        torch.clamp(self.A_s, *self.bounds["A"])
        torch.clamp(self.sigma_1, *self.bounds["sigma"])
        torch.clamp(self.sigma_2, *self.bounds["sigma"])

    def get_dog_parameters(self, i: int = 0) -> dict:
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

    def plot_dog(self, n: int = None) -> None:
        """plot first n dog unit filters

        Args:
            n (int, optional): number of filters to plot, if None plots all.
                Defaults to None.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if n == None:
            n = self.num_lgn_units
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dogs = self.compute_dog()
        x = torch.tensordot(x.float(), dogs.float(), dims=((2, 3), (1, 2)))
        return x


class HSMCorticalActivation(HSMLayers):
    """The activation (non-linearity) part of a cortical layer as described in
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927"""

    def __init__(self, input_size: int, *args, **kwargs) -> None:
        """
        Args:
            input_size (int): input size in the activation layer.
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.thresholds = nn.Parameter(
            torch.empty(
                size=(1, self.input_size),
                dtype=torch.float,
                device=self.device,
            ).uniform_()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log10(1 + torch.exp(x - self.thresholds))
        return x


class HSMCortialFullyConnected(HSMLayers):
    """The fully connected part of a cortical layer as decribed in
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927"""

    def __init__(self, input_size: int, output_size: int, *args, **kwargs) -> None:
        """
        Args:
            input_size (int): input size (number of units) in the fully connected
                layer
            output_size (int): output size (number of units) of the fully connected
                layer
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.fully_connected = nn.Linear(
            in_features=self.input_size,
            out_features=self.output_size,
            device=self.device,
            dtype=torch.float,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_connected(x)
        return x


class HSMCorticalBlock(HSMLayers):
    """A Cortical Block as described in
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927
    """

    def __init__(self, input_size: int, output_size: int, *args, **kwargs) -> None:
        """
        Args:
            input_size (int): input size of the layer. In the first cortical layer
                this is num_lgn_units and in the second cortical layer this is
                hidden_units_fraction * num_neurons.
            output_size (int): output size of the lay. In the first cortical layer
                this is hidden_units_fraction * num_neurons and in the second cortical
                layer this is num_neurons.
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.block = nn.Sequential(
            HSMCortialFullyConnected(
                input_size=self.input_size,
                output_size=self.output_size,
                *args,
                **kwargs,
            ),
            HSMCorticalActivation(input_size=self.output_size, *args, **kwargs),
        )
        # self.layer = HSMCortialFullyConnected(
        #     input_size=self.input_size, output_size=self.output_size, *args, **kwargs
        # )
        # self.activation = HSMCorticalActivation(
        #     input_size=self.output_size, *args, **kwargs
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        # x = self.layer(x)
        # x = self.activation(x)
        return x


#%%
#
