"""Module providing generator loss functions."""

import torch
import torch.nn as nn


class SelectedNeuronActivation(nn.Module):
    """Naive loss function which maximizes the selected neuron's activation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, activations: torch.Tensor, neuron_idx: int) -> torch.Tensor:
        """Return the scalar neuron activation of the selected neuron.

        Args:
            activations (torch.Tensor): input activation tensor.
            neuron_idx (int): neuron to select

        Returns:
            torch.Tensor: output
        """
        average_activation = activations[
            :, neuron_idx
        ].sum()  # .div(activations.shape[0])
        loss = -average_activation
        return loss


class SelectedNeuronActivityWithDiffernceInImage(nn.Module):
    """Loss function which maximizes neural activation and difference_scaling * difference in images."""

    def __init__(self, difference_scaling: float = 0.01) -> None:
        """Instantiate Loss function.

        The difference_scaling factor scales the effect of images on the loss to
        be <difference_scaling> * the effect the activation has on the loss.

        Args:
            difference_scaling (float, optional): Scaling term. Defaults to 0.01.
        """
        super().__init__()
        self.difference_scaling = difference_scaling
        assert self.difference_scaling <= 1, (
            "If the effect of the image difference on the loss is more than"
            " the effect of the activation, the generator no longer trains "
            "to maximize the activation, but rather to make different images."
            " This is not useful!"
        )

    def forward(
        self, activations: torch.Tensor, images: torch.Tensor, neuron_idx: int
    ) -> torch.Tensor:
        """Return the scalar neuron activation of the selected neuron with a
        regularization term.

        The regularization term is the sum over pixel differences in images.

        Args:
            activations (torch.Tensor): input activation tensor.
            images (torch.Tensor): input image tensor. Image, which is passed to encoding model.
            neuron_idx (int): neuron to select

        Returns:
            torch.Tensor: output
        """
        average_activation = activations[
            :, neuron_idx
        ].sum()  # .div(activations.shape[0])
        img_difference = images.diff(dim=0).sum()
        autoscaling = average_activation / img_difference
        loss = -(
            average_activation + self.difference_scaling * autoscaling * img_difference
        )
        return loss


if __name__ == "__main__":
    pass
