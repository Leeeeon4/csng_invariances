#%%
import torch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from typing import Tuple, List

#%%


def cluster_generated_images(
    images: torch.Tensor,
    activations: torch.Tensor,
    selected_neuron_idx: int,
    num_representative_samples: int = 6,
    show: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    flattend_images = images.reshape(
        images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
    )

    clusters = torch.from_numpy(
        AgglomerativeClustering(
            n_clusters=num_representative_samples, affinity="cosine", linkage="complete"
        ).fit_predict(flattend_images)
    )

    sorted_images = []
    sorted_activations = []
    for i in range(num_representative_samples):
        print(i)
        print((clusters == i).sum())
        sorted_images.append(images[clusters == i, :, :, :])
        sorted_activations.append(activations[clusters == i, :])

    if show:
        for i in range(num_representative_samples):
            imgs = sorted_images[i]
            actis = sorted_activations[i]
            for j in range(imgs.shape[0]):
                plt.imshow(imgs[j, :, :, :].squeeze(), cmap="gray")
                plt.title(
                    f"Cluster {i}, image {j} - Activation {actis[j,selected_neuron_idx]}"
                )
                img_title = f"Neuron_{selected_neuron_idx}_cluster_{i}_image_{j}.png"
                plt.savefig()
                plt.close()
    return sorted_images, sorted_activations


# %%
show = True
num_representative_samples = 6

images = torch.randint(low=0, high=255, size=(64, 1, 36, 64))
activations = torch.rand(size=(64, 25))

clustered_images, clustered_activations = cluster_generated_images()
#%%
def cluster_generated_images(
    images: torch.Tensor,
    activations: torch.Tensor,
    selected_neuron_idx: int,
    num_representative_samples: int = 6,
    show: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    flattend_images = images.reshape(
        images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
    )

    clusters = torch.from_numpy(
        AgglomerativeClustering(
            n_clusters=num_representative_samples, affinity="cosine", linkage="complete"
        ).fit_predict(flattend_images)
    )

    sorted_images = []
    sorted_activations = []
    for i in range(num_representative_samples):
        print(i)
        print((clusters == i).sum())
        sorted_images.append(images[clusters == i, :, :, :])
        sorted_activations.append(activations[clusters == i, :])

    if show:
        for i in range(num_representative_samples):
            imgs = sorted_images[i]
            actis = sorted_activations[i]
            for j in range(imgs.shape[0]):
                plt.imshow(imgs[j, :, :, :].squeeze(), cmap="gray")
                plt.title(
                    f"Cluster {i}, image {j} - Activation {actis[j,selected_neuron_idx]}"
                )
                img_title = f"Neuron_{selected_neuron_idx}_cluster_{i}_image_{j}.png"
                plt.savefig()
                plt.close()
    return sorted_images, sorted_activations


# %%
