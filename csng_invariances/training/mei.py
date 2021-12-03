"""Module providing Most Exciting Image (MEI) training."""

from pathlib import Path
import torch
from rich.progress import track
import matplotlib.pyplot as plt
from numpy import save

from csng_invariances._utils.utlis import string_time


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
    show_last = True
    meis = {}
    t = string_time()
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
            description=f"Neuron {counter+1}/{len(selected_neuron_indicies)}: ",
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
        save(
            meis_directoy / f"MEI_neuron_{neuron}.npy", new_image.detach().cpu().numpy()
        )
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
            im = ax.imshow(new_image.detach().cpu().numpy().squeeze())
            ax.set_title(f"Final image neuron {neuron}")
            plt.colorbar(im)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(3)
            plt.close("all")
        if show_last:
            img = new_image.detach().cpu().numpy().squeeze()
            activations = encoding_model(new_image)
            plt.imshow(img, cmap="gray")
            plt.colorbar()
            plt.title(f"Activation: {activations[:,neuron].item()}")
            image_title = f"mei_neuron_{neuron}.png"
            image_directory = Path.cwd() / "reports" / "figures" / "mei" / t
            image_directory.mkdir(parents=True, exist_ok=True)
            plt.savefig(image_directory / image_title, facecolor="white")
            plt.close()
    return meis
