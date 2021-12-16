"""Submodule for computing single neuron correlations."""

import torch
from rich.progress import track
from numpy import corrcoef, save, load
from pandas import read_csv
from csng_invariances._utils.utlis import string_time
from csng_invariances.models.linear_filter import Filter
from pathlib import Path

###########################DNN ENCODING MODEL###########################
def compute_single_neuron_correlations_encoding_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    responses: torch.Tensor,
    batch_size: int = 64,
    **kwargs: int,
) -> torch.Tensor:
    """Computes the single neuron corrlations.

    Args:
        model (nn.Module): Encoding model to use for predictions
        images (Tensor): image tensor of all images, not batched.
        responses (Tensor): response tensor of all images, not batched.
        batch_size (int, optional): batch_size to use for computation. Defaults to 64.

    Returns:
        Tensor: single neuron correlations tensor of dimension (neuron_count,
            image_count)
    """
    neuron_count = responses.shape[1]
    image_count = responses.shape[0]
    num_batches, num_images_last_batch = divmod(image_count, batch_size)

    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    single_neuron_correlations = torch.empty(neuron_count, device=device)
    predictions = torch.empty_like(responses)
    with torch.no_grad():
        for batch in track(
            range(num_batches), description="Computing encoding model correlation"
        ):
            image_batch = images[
                (batch) * batch_size : (batch + 1) * batch_size, :, :, :
            ]
            prediction_batch = model(image_batch.to(device))
            predictions[
                (batch) * batch_size : (batch + 1) * batch_size, :
            ] = prediction_batch
        image_last_batch = images[
            (num_batches * batch_size) : (
                num_batches * batch_size + num_images_last_batch
            ),
            :,
            :,
            :,
        ]
        prediction_last_batch = model(image_last_batch.to(device))
        predictions[
            (num_batches * batch_size) : (
                num_batches * batch_size + num_images_last_batch
            ),
            :,
        ] = prediction_last_batch
    for neuron in track(
        range(neuron_count),
        description="Finishing tensor build of single neuron correlations.",
    ):
        single_neuron_correlation = corrcoef(
            responses[:, neuron].cpu(), predictions[:, neuron].cpu()
        )[0, 1]
        single_neuron_correlations[neuron] = single_neuron_correlation

    # TODO Optional: Make a structured saving function, pass list of directory, filename, and readme string.
    t = string_time()
    directory = Path.cwd() / "reports" / "encoding" / "single_neuron_correlations" / t
    directory.mkdir(parents=True, exist_ok=True)
    save(
        directory / "single_neuron_correlations.npy",
        single_neuron_correlations.cpu().numpy(),
    )
    if (directory / "readme.md").exists() is False:
        with open(directory / "readme.md", "w") as f:
            f.write(
                "# Readme.md\n"
                f"Single neuron correlations are stored in 'singel_neuron_correlations.npy'. "
                f"The order of neurons is the same as in the original data. Dimensions "
                f"are: {single_neuron_correlation.shape}."
            )
    print(f"The encoding model single neuron correlataions are stored in:\n{directory}")
    return single_neuron_correlations


def load_single_neuron_correlations_encoding_model(
    path: str, device: str = None
) -> torch.Tensor:
    """Load single neuron correlation tensor

    Args:
        path (str): path to file.
        device (str, optional): torch.device, if None is passed, try to use cuda.
            Defaults to None.

    Returns:
        torch.Tensor: single_neuron_correlation tensor of shape (num_neurons) in
            original dataorder on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data = load(path)
        data_tensor = torch.from_numpy(data)
        data_tensor = data_tensor.to(device)
    except Exception:
        print(
            "An error occured. The file could not be loaded. Is the path correct? "
            "Is it a numpy file?"
        )
    return data_tensor


###########################Linear Filter###########################
def compute_single_neuron_correlations_linear_filter(
    train_filter: Filter, device: str = None
) -> torch.Tensor:
    """Return single neuron correlations.

    Single neuron correlations are computed in every Filter.predict() step.

    Args:
        train_filter (Filter): Trained filter after hyperparametersearch is conducted.
        device (str, optional): torch device if None, tries cuda. Defaults to None.

    Returns:
        torch.Tensor: single neuron correlations in original data order on device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_neuron_correlations_linear_filter = torch.from_numpy(
        train_filter.single_neural_correlation_linear_filter
    )
    single_neuron_correlations_linear_filter.to(device)
    return single_neuron_correlations_linear_filter


def load_single_neuron_correlations_linear_filter(
    path: str, device: str = None
) -> torch.Tensor:
    """Load single neuron correlations Tensor.

    Args:
        path (str): path to numpy file.
        device (str, optional): torch device, if None, tries cuda. Defaults to None.

    Returns:
        torch.Tensor: single_neuron_correlations Tensor in original data order on
            device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if Path(path).suffix == ".csv":
            csv = read_csv(Path(path))
            data = [float(csv.columns[1])]
            for i in csv.iloc(axis=1)[1].to_list():
                data.append(i)
            data_tensor = torch.tensor(data, device=device)
        elif Path(path).suffix == ".npy":
            data = load(path)
            data_tensor = torch.from_numpy(data)
            data_tensor = data_tensor.to(device)
    except Exception as err:
        print(
            "An error occured. The file could not be loaded. Is the path correct? "
            "Is it either a csv or a npy file?"
        )
        print(err)
    return data_tensor


if __name__ == "__main__":
    pass
