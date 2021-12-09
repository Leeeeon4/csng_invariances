"""Submodule providing different clustering approaches."""

import torch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from typing import Tuple, List
from pathlib import Path
from csng_invariances._utils.utlis import string_time


def cluster_generated_images(
    images: torch.Tensor,
    activations: torch.Tensor,
    selected_neuron_idx: int,
    num_clusters: int = 8,
    show: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Cluster generated images.

    Args:
        images (torch.Tensor): generated images.
        activations (torch.Tensor): activations of generated images
        selected_neuron_idx (int): current neuron.
        num_representative_samples (int, optional): Number of clusters. Defaults to 6.
        show (bool, optional): If true, every image is stored. Defaults to False.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Tuple of lists clustered_images and clustered_activations
    """
    flattend_images = (
        images.reshape(
            images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
        )
        .cpu()
        .numpy()
    )
    t = string_time()

    clusters = torch.from_numpy(
        AgglomerativeClustering(
            n_clusters=num_clusters, affinity="cosine", linkage="complete"
        ).fit_predict(flattend_images)
    )

    sorted_images = []
    sorted_activations = []
    for i in range(num_clusters):
        sorted_images.append(images[clusters == i, :, :, :])
        sorted_activations.append(activations[clusters == i])

    if show:
        for i in range(num_clusters):
            imgs = sorted_images[i]
            actis = sorted_activations[i]
            for j in range(imgs.shape[0]):
                plt.imshow(imgs[j, :, :, :].squeeze(), cmap="gray")
                plt.title(
                    f"Cluster {i}, image {j} - Activation {actis[j,selected_neuron_idx]}"
                )
                img_title = f"Neuron_{selected_neuron_idx}_cluster_{i}_image_{j}.png"
                img_path = (
                    Path.cwd()
                    / "reports"
                    / "figures"
                    / "generator"
                    / "after_training"
                    / "clustered"
                    / str(selected_neuron_idx)
                    / t
                )
                img_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(img_path / img_title)
                plt.close()

    return sorted_images, sorted_activations
