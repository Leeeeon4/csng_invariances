"""Module providing Most Exciting Image (MEI) funcitonality.

Concept: 
1. preprocess images if necessary.
_, c, w, h = images.shape
2. src_image = torch.zeros(1, c, w, h, requires_grad=True, device=device)
"""

from pathlib import Path
from datetime import datetime
import torch
from rich.progress import track
import matplotlib.pyplot as plt
from numpy import save


def naive_gradient_ascent_step(
    criterion: torch.nn.Module,
    encoding_model: torch.nn.Module,
    image: torch.Tensor,
    neuron_idx: int,
    lr: float = 0.01,
    *args: int,
    **kwargs: int,
) -> torch.Tensor:
    """Performs one step of naive gradient ascent.

    Adds the scaled gradient to the image and return it.

    Args:
        criterion (torch.nn.Module): loss function a.k.a. criterion.
        encoding_model (torch.nn.Module): trained encoding model which is the
            basis for the meis
        image (torch.Tensor): gaussian white noise image in pytorch convention.
        neuron_idx (int): neuron to analyize
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        torch.Tensor: image after one gradient ascent step.
    """
    # Image normalization
    with torch.no_grad():
        image /= image.max()
    loss = criterion(encoding_model(image), neuron_idx)
    loss.backward()
    # Gradient ascent step.
    # The scaled and then multiplied by the learning rate.
    with torch.no_grad():
        image += image.grad * image.max() / image.grad.max() * lr
    image.grad = None
    return image


def mei(
    criterion: torch.nn.Module,
    encoding_model: torch.nn.Module,
    image: torch.Tensor,
    selected_neuron_indicies: list,
    lr: float = None,
    epochs: int = 200,
    show: bool = False,
    *args: int,
    **kwargs: int,
) -> dict:
    """Compute the MEIs.

    Computes the MEIs by means of gradient ascent for the selected_neuron_indicies list and stores them.

    Args:
        criterion (torch.nn.Module): Loss function
        encoding_model (torch.nn.Module): encoding model which is basis for MEIs
        image (torch.Tensor): Gaussian white noise image
        selected_neuron_indicies (list): List of neurons to compute MEIs for
        lr (float, optional): Learning Rate. Defaults to None.
        epochs (int, optional): Number of epochs. Defaults to 200.
        show (bool, optional): If True, MEIs are shown after every 10 steps.
            Defaults to False.

    Returns:
        dict: Dictionary of neuron_idx and the associated MEI.
    """
    meis = {}
    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    meis_directoy = Path.cwd() / "data" / "processed" / "MEIs" / t
    meis_directoy.mkdir(parents=True, exist_ok=True)
    for counter, neuron in enumerate(selected_neuron_indicies):
        trainer_config = {
            "criterion": criterion,
            "encoding_model": encoding_model,
            "epochs": epochs,
            "neuron_idx": neuron,
        }
        if lr is not None:
            trainer_config["lr"] = lr
        old_image = image.detach().clone()
        old_image.requires_grad = True
        new_image = naive_gradient_ascent_step(image=old_image, **trainer_config)
        for epoch in track(
            range(epochs),
            total=epochs,
            description=f"Neuron {counter}/{len(selected_neuron_indicies)}: ",
        ):
            new_image = naive_gradient_ascent_step(image=new_image, **trainer_config)
            if show and epoch % 10 == 0:
                fig, ax = plt.subplots(figsize=(6.4 / 2, 3.6 / 2))
                im = ax.imshow(new_image.detach().numpy().squeeze())
                ax.set_title(f"Image neuron {neuron} after {epoch} epochs")
                plt.colorbar(im)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
        save(meis_directoy / f"MEI_neuron_{neuron}.npy", new_image.detach().numpy())
        if (meis_directoy / "readme.md").exists() is False:
            with open(meis_directoy / "readme.md", "w") as f:
                f.write(
                    "# readme\n"
                    "The MEIs (Most Exciting Images) store the pytorch 4D representation"
                    "of the image per neuron. Dimensions are (batchsize, channels, "
                    "height, width)."
                )
        meis[neuron] = new_image
        if show:
            fig, ax = plt.subplots(figsize=(6.4 / 2, 3.6 / 2))
            im = ax.imshow(new_image.detach().numpy().squeeze())
            ax.set_title(f"Final image neuron {neuron}")
            plt.colorbar(im)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(3)
            plt.close("all")
    return meis
