"""Module with misc. function for data handeling."""


import numpy as np
import torch
import torch.utils.data as utils
import json

from pathlib import Path
from torchvision import transforms

from csng_invariances._neuralpredictors.data.samplers import RepeatsBatchSampler


# Testdata
def get_test_dataset():
    """Return small dataset for testing.

    Returns:
        tuple: Tuple of train_images, train_responses, val_images and val_responses.
    """
    train_images = torch.from_numpy(np.random.randn(100, 1, 12, 13))
    train_responses = torch.from_numpy(np.random.randn(100, 14))
    val_images = torch.from_numpy(np.random.randn(20, 1, 12, 13))
    val_responses = torch.from_numpy(np.random.randn(20, 14))
    return train_images, train_responses, val_images, val_responses


def handle_numpy_torch_tensor(func):
    """Automatically handle torch and numpy tensors differently.

    Args:
        func (function): Function to handle

    Returns:
        tensor:
            type numpy tensor if a numpy tensor was passed.
            type torch tensor if a torch tensor was passed.

    """

    def wrapper(*args, **kwargs):
        if args:
            if torch.is_tensor(*args):
                tensor = func(*args)
            else:
                tensor = torch.from_numpy(*args)
                tensor = func(tensor)
                tensor = tensor.numpy()
        if kwargs:
            if torch.is_tensor(list(**kwargs.values())[0]):
                tensor = func(list(**kwargs.values())[0])
            else:
                tensor = torch.from_numpy(list(**kwargs.values())[0])
                tensor = func(tensor)
                tensor = tensor.numpy()
        return tensor

    return wrapper


@handle_numpy_torch_tensor
def scale_tensor_to_0_1(tensor):
    """Scales a tensor to values between 0 and 1.

    Args:
        tensor (torch.tensor): Input tensor.

    Returns:
        Tensor: Scaled input tensor.
    """
    tensor = tensor.add(abs(tensor.min()))
    tensor = tensor.div(tensor.max())
    return tensor


@handle_numpy_torch_tensor
def normalize_tensor_zero_mean_unit_standard_deviation(tensor):
    """Normalize tensor to zero mean and unit standard deviation.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized input tensor.
    """
    tensor = transforms.Normalize(0, 1)(tensor)
    return tensor


@handle_numpy_torch_tensor
def normalize_tensor_by_standard_deviation_devision(tensor):
    """Normalize tensor by dividing thorugh its standard deviation.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized input tensor.
    """
    mean = torch.std(tensor)
    tensor = tensor.div(mean)
    return tensor


def make_directories():
    """Makes basic directory structure expected.

    Returns:
        list: List of created directories.
    """
    data_dir = Path.cwd() / "data"
    data_external_dir = data_dir / "external"
    data_interim = data_dir / "interim"
    data_processed = data_dir / "processed"
    data_raw = data_dir / "raw"
    models_dir = Path.cwd() / "models"
    models_external_dir = models_dir / "external"
    report_dir = Path.cwd() / "reports"
    report_figure_dir = report_dir / "figures"
    directories = [
        data_dir,
        data_external_dir,
        data_interim,
        data_processed,
        data_raw,
        models_dir,
        models_external_dir,
        report_dir,
        report_figure_dir,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    return directories


def get_oracle_dataloader(
    dat, toy_data=False, oracle_condition=None, verbose=False, file_tree=False
):
    """This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX."""
    if toy_data:
        condition_hashes = dat.info.condition_hash
    else:
        dat_info = dat.info if not file_tree else dat.trial_info
        if "image_id" in dir(dat_info):
            condition_hashes = dat_info.image_id
            image_class = dat_info.image_class

        elif "colorframeprojector_image_id" in dir(dat_info):
            condition_hashes = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif "frame_image_id" in dir(dat_info):
            condition_hashes = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError(
                "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                "in order to load get the oracle repeats."
            )

    max_idx = condition_hashes.max() + 1
    classes, class_idx = np.unique(image_class, return_inverse=True)
    identifiers = condition_hashes + class_idx * max_idx

    dat_tiers = dat.tiers if not file_tree else dat.trial_info.tiers
    sampling_condition = (
        np.where(dat_tiers == "test")[0]
        if oracle_condition is None
        else np.where((dat_tiers == "test") & (class_idx == oracle_condition))[0]
    )
    if (oracle_condition is not None) and verbose:
        print("Created Testloader for image class {}".format(classes[oracle_condition]))

    sampler = RepeatsBatchSampler(identifiers, sampling_condition)
    return utils.DataLoader(dat, batch_sampler=sampler)


def unpack_data_info(data_info):
    """This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX."""
    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels


def save_configs(configs, model_directory):
    """Save configurations dicts as json.

    Args:
        configs (dict): Dictionary of sub_config dicts.
        model_directory (path): path to directory in which model is stored.
    """
    configs["trainer_config"]["device"] = str(configs["trainer_config"]["device"])
    for key, value in configs.items():
        with open(model_directory / f"{key}.json", "w") as outfile:
            json.dump(value, outfile, indent=2)


def load_configs(model_directory):
    """Load configs in given directory.

    Args:
        model_directory (path): Path to directory in which model is stored.

    Returns:
        dict: Dictionary of sub_config dicts.
    """
    configs = {}
    for file in Path(model_directory).iterdir():
        if file.suffix == ".json":
            with open(file, "r") as content:
                configs[file.stem] = json.load(content)
    configs["trainer_config"]["device"] = torch.device(
        "cuda" if configs["trainer_config"]["device"] == "cuda" else "cpu"
    )
    return configs
