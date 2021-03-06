"""Module from preprocessing of images and responses."""

from csng_invariances.data._data_helpers import *


def image_preprocessing(images: torch.Tensor) -> torch.Tensor:
    """Preprocesses images

    Args:
        images (Tensor): Image tensor.

    Returns:
        Tensor: Preprocessed images.
    """
    images = normalize_tensor_zero_mean_unit_standard_deviation(images)
    images = scale_tensor_to_0_1(images)
    return images


def response_preprocessing(responses: torch.Tensor) -> torch.Tensor:
    """Preprocesses responses

    Args:
        responses (Tensor): Response tensor.

    Returns:
        Tensor: Preprocessed responses
    """
    # responses = normalize_tensor_by_standard_deviation_devision(responses)
    return responses
