import numpy as np
import torch.utils.data as utils
from pathlib import Path
from neuralpredictors.data.samplers import RepeatsBatchSampler


def normalize_tensor_to_0_1(tensor):
    """Normalizes a tensor to values between 0 and 1.

    Args:
        tensor (torch.tensor): Input tensor.

    Returns:
        torch.tensor: Input tensor normalized to 0 and 1.
    """
    return tensor.div(tensor.max())


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
