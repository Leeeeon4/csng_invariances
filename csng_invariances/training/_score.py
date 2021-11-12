"""Provide different utilies for computing the neuron scores.

The score is basis for selecting the neurons to examine in further experiments.
"""

import numpy as np


def sta_predictions(images, sta):
    """Compute estimate of receptive field based on Spike-triggered Average.

    Computes a linear estimate of the receptive field of the neurons based on
    the Spike-triggered Average (sta). This is feasable for simple cells.

    Args:
        images (np.array): 2D representation of image data. Images are expected
            to be passed flattened.
        sta (np.array): 2D representation of sta. Images are still flattened.

    Returns:
        np.array: vector of linear estimation of receptive field of
            neurons based on sta.
    """
    sta_prediction = np.matmul(images.T, sta)
    return sta_prediction


def sta_correlations(sta_predictions, responses):
    """Compute correlation of sta prediction to real responses

    Args:
        sta_predictions (np.array): Vector of sta predictions.
        responses (np.array): Vector of neural responses.

    Returns:
        np.array: Vector of correlations of sta_predictions to response data.
    """
    sta_corr = np.corrcoef(sta_predictions, responses)
    return sta_corr


def discriminator_correlations(discriminator_predictions, responses):
    """Compute correlation of discriminator prediction to real responses

    Args:
        discriminator_predictions (np.array): Vector of
        responses (np.array): Vector of neural responses

    Returns:
        np.array: Vector of correlations of discriminator_predictions to
            response data.
    """
    discriminator_corr = np.corrcoef(discriminator_predictions, responses)
    return discriminator_corr


def selection_score(sta_corr, discriminator_corr):
    """Compute neuron selection score.

    Args:
        sta_corr (np.array): Vector of correlations of sta_predictions to
            response data.
        discriminator_corr (np.array): Vector of correlations of
            discriminator_predictions to response data.

    Returns:
        np.array: Vector of scores-values for neurons.
    """
    score = (1-sta_corr)*discriminator_corr
    return score
