#%%
from typing import List
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from rich import print
from numpy import load
from csng_invariances.metrics_statistics.correlations import (
    load_single_neuron_correlations_encoding_model,
    load_single_neuron_correlations_linear_filter,
)
from csng_invariances.models.encoding import load_encoding_model
from csng_invariances.data._data_helpers import scale_tensor_to_0_1
from csng_invariances.models.generator import (
    FullyConnectedGeneratorWithGaussianBlurring,
)
from csng_invariances.data.datasets import normal_latent_vector, uniform_latent_vector

# %%
def plot_neuron_x_with_8_clusters(
    selected_neuron_idx: int,
    lrf: torch.Tensor,
    meis: dict,
    roi: torch.Tensor,
    mask: torch.Tensor,
    epochs_images: torch.Tensor,
    clustered_images: List[torch.Tensor],
    clustered_activations: List[torch.Tensor],
    score: torch.Tensor,
    lrf_correlations: torch.Tensor,
    enc_correlations: torch.Tensor,
    training_data: List[torch.Tensor],
    generation_data: List[torch.Tensor],
    encoding_model: torch.nn.Module,
    generator_model: torch.nn.Module,
    config: dict,
    *args,
    **kwargs,
) -> None:
    """Plot a neuron report and saves it.

    Args:
        selected_neuron_idx (int): current neuron
        lrf (torch.Tensor): tensor of linear receptive field as estimated by
            linear filter (<neuron>, <channel>, <height>, <width>)
        meis (dict): dictionary with key: <neuron> and value: <mei of neuron>,
            where mei of neuron (<neuron>, <channel>, <height>, <width>)
        roi (torch.Tensor): tensor of region of interest (<neuron>, <channel>,
            <height>, <width>)
        mask (torch.Tensor): tensor of mask (<neuron>, <channel>, <height>, <width>)
        epochs_images (torch.Tensor): tensor of image
        clustered_images (List[torch.Tensor]): [description]
        clustered_activations (List[torch.Tensor]): [description]
        score (torch.Tensor): [description]
        lrf_correlations (torch.Tensor): [description]
        enc_correlations (torch.Tensor): [description]
        training_data (List[torch.Tensor]): [description]
        generation_data (List[torch.Tensor]): [description]
        encoding_model (torch.nn.Module): [description]
        generator_model (torch.nn.Module): [description]
        config (dict): [description]
    """
    axis_off = True
    num_clusters = len(clustered_images)
    num_epochs = epochs_images.shape[0]
    one_lrf = scale_tensor_to_0_1(
        lrf[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
    )
    one_roi = scale_tensor_to_0_1(
        roi[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
    )
    if mask is not None:
        one_mask = scale_tensor_to_0_1(
            mask[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
        )

    for key in meis.keys():
        if type(key) is str:
            if type(selected_neuron_idx) is str:
                neuron = selected_neuron_idx
            else:
                neuron = str(selected_neuron_idx)

    one_mei = scale_tensor_to_0_1(meis[neuron].detach().cpu().squeeze())
    one_mei_activation = encoding_model(
        meis[neuron].reshape(1, 1, one_mei.shape[0], one_mei.shape[1])
    )[:, selected_neuron_idx].item()

    epochs_slicer = [i * (int(num_epochs / 4)) for i in range(4)]

    epochs_tensor_list = [
        scale_tensor_to_0_1(epochs_images[i, :, :, :].detach().cpu().squeeze())
        for i in epochs_slicer
    ]

    clusters_tensor_list = [
        scale_tensor_to_0_1(i[0, :, :, :].detach().cpu().squeeze())
        for i in clustered_images
    ]
    clusters_activation_list = [i[0].item() for i in clustered_activations]

    fig, axes = plt.subplots(nrows=12, ncols=4, figsize=(8.27, 11.70))

    gs = axes[0, 0].get_gridspec()
    large_axes = []
    for row in [0, 1, 4, 7]:
        for col in axes[row, :]:
            col.remove()
        large_axes.append(fig.add_subplot(gs[row, :]))

    if axis_off:
        for row in range(12):
            for col in range(4):
                axes[row][col].axis("off")
        for ax in large_axes:
            ax.axis("off")

    # Row 0
    large_axes[0].text(
        0.5,
        0.5,
        f"Neuron {selected_neuron_idx}",
        fontsize=24,
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Row 1
    large_axes[1].text(
        0,
        1,
        (
            r"All images are scaled to be $\in$ [0, 1] for visualization."
            f"\nThe encoding model used was {encoding_model.__class__.__name__}."
            f"\nNeuron score is: {score[selected_neuron_idx]:.04f} | LRF correlation is {lrf_correlations[selected_neuron_idx]:.04f} | Encoding model correlation is {enc_correlations[selected_neuron_idx]:.04f}"
        ),
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Row 2
    im_2_0 = axes[2][0].imshow(one_lrf)
    axes[2][0].set_title("LRF", fontsize=10)

    im_2_1 = axes[2][1].imshow(one_roi)
    axes[2][1].set_title("ROI", fontsize=10)

    if mask is not None:
        im_2_2 = axes[2][2].imshow(one_mask, cmap="gray")
        axes[2][2].set_title("Mask", fontsize=10)

    im_2_3 = axes[2][3].imshow(one_mei)
    axes[2][3].set_title(f"MEI | {one_mei_activation:.04f}", fontsize=10)

    # Row 3
    axes[3][0].text(
        0,
        1,
        (
            r"$\mu_{luminance}$: "
            f"{one_lrf.mean():.04f}\n"
            r"$\sigma_{luminance}$: "
            f"{one_lrf.std():.04f}"
        ),
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    axes[3][3].text(
        0,
        1,
        (
            r"$\mu_{luminance}$: "
            f"{one_mei.mean():.04f}\n"
            r"$\sigma_{luminance}$: "
            f"{one_mei.std():.04f}"
        ),
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Row 4
    large_axes[2].text(
        0,
        1,
        (
            f"Generator model used was {generator_model.__class__.__name__}.\n"
            f"Model was trained on {len(training_data)} batches of size {training_data[0].shape[0], training_data[0].shape[1]} drawn from N~(0, 1).\n"
            f"Images were generated on {len(generation_data)} batches drawn from U~[-2, 2]."
        ),
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Row 5 & 6
    for counter, (image, epoch) in enumerate(zip(epochs_tensor_list, epochs_slicer)):
        axes[5][counter].imshow(image)
        axes[5][counter].set_title(f"Epoch: {epoch}", fontsize=10)
        axes[6][counter].text(
            0,
            1,
            (
                r"$\mu_{luminance}$: "
                f"{image.mean():.04f}\n"
                r"$\sigma_{luminance}$: "
                f"{image.std():.04f}"
            ),
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="top",
        )

    # Row 7
    large_axes[3].text(
        0,
        1,
        (
            "Generated images are clustered by AgglomerativeClustering into 8 clusters.\n"
            "Thus, the eigth 'most different' images are presented.\n"
            "One image is presented for each cluster."
        ),
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Row 8 & 9
    for counter, (image, activation) in enumerate(
        zip(clusters_tensor_list[0:4], clusters_activation_list[0:4])
    ):
        axes[8][counter].imshow(image)
        axes[8][counter].set_title(
            f"Cluster {counter} | {activation:.04f}", fontsize=10
        )
        axes[9][counter].text(
            0,
            1,
            (
                r"$\mu_{luminance}$: "
                f"{image.mean():.04f}\n"
                r"$\sigma_{luminance}$: "
                f"{image.std():.04f}"
            ),
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="top",
        )

    # Row 10 & 11
    for counter, (image, activation) in enumerate(
        zip(clusters_tensor_list[4:], clusters_activation_list[4:])
    ):
        axes[10][counter].imshow(image)
        axes[10][counter].set_title(
            f"Cluster {counter+4} | {activation:.04f}", fontsize=10
        )
        axes[11][counter].text(
            0,
            0,
            (
                r"$\mu_{luminance}$: "
                f"{image.mean():.04f}\n"
                r"$\sigma_{luminance}$: "
                f"{image.std():.04f}"
            ),
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    directory = (
        Path.cwd()
        / "reports"
        / "overview"
        / f"{config['Timestamp']}_{config['wandb_name']}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(directory / f"Overview_neuron_{selected_neuron_idx}.png")


# %%
# %%
def plot_examples_of_generated_images(
    selected_neuron_idx: int,
    batch_counter: int,
    sample_image: torch.Tensor,
    masked_image: torch.Tensor,
    preprocessed_image: torch.Tensor,
    encoding_model: torch.nn.Module,
    image_directory: Path,
    *args,
    **kwargs,
) -> None:
    """Plots one example of generated images per batch.

    Images are the sample_image, the masked_image and the preprocessed_image

    Args:
        selected_neuron_idx (int): current neuron
        batch_counter (int): current batch of generation
        sample_image (torch.Tensor): sample_image
        masked_image (torch.Tensor): masked_image
        preprocessed_image (torch.Tensor): preprocessed_image
        encoding_model (torch.nn.Module): encoding model
        image_directory (Path): image directory.
    """
    plt.imshow(sample_image[0, :, :, :].detach().cpu().numpy().squeeze())
    plt.title(f"Batch {batch_counter:02d}")
    plt.colorbar()
    plt.savefig(
        image_directory
        / f"{batch_counter:02d}_01_First_sample_in_batch_{batch_counter}.jpg"
    )
    plt.close()
    plt.imshow(masked_image[0, :, :, :].detach().cpu().numpy().squeeze())
    plt.title(f"Batch {batch_counter:02d}")
    plt.colorbar()
    plt.savefig(
        image_directory
        / f"{batch_counter:02d}_02_First_masked_sample_in_batch_{batch_counter}.jpg"
    )
    plt.close()
    activation = encoding_model(preprocessed_image)
    plt.imshow(preprocessed_image[0, :, :, :].detach().cpu().numpy().squeeze())
    plt.title(
        f"Batch {batch_counter:02d} | Activation: {activation[0,selected_neuron_idx].item():.3f}"
    )
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(
        image_directory
        / f"{batch_counter:02d}_03_First_preprocessed_sample_in_batch_{batch_counter}.jpg"
    )
    plt.close()


if __name__ == "__main__":
    #%%
    selected_neuron_idx = 1706
    num_clusters = 8

    # tensor [neurons, channels, height, width]
    lrf = torch.from_numpy(
        load(
            "/home/leon/csng_invariances/data/processed/linear_filter/2021-11-30_15:15:09/evaluated_filter.npy"
        )
    )
    one_lrf = lrf[selected_neuron_idx, :, :, :].reshape(1, 1, 36, 64)
    print(f"lrf: {one_lrf.shape}")
    # TODO tensor(neurons, neurons) makes sense????
    lrf_activation = torch.randn(5335, 5335)

    # dict of len (selected_neuron_idxs) with neuron:tensor(1, channels, height, width)
    meis = {}
    mei = torch.from_numpy(
        load(
            "/home/leon/csng_invariances/data/processed/MEIs/2021-12-02_15:46:31/MEI_neuron_1706.npy"
        )
    )
    meis[selected_neuron_idx] = mei
    # dict of len (selected_neuron_idxs) with neuron:activation
    mei_activations = {}
    mei_activation = torch.randn(1)
    mei_activations[selected_neuron_idx] = mei_activation
    print(f"mei: {mei.shape}")

    # tensor (neuron, channel, height, width)
    roi = torch.from_numpy(
        load(
            "/home/leon/csng_invariances/data/processed/roi/2021-11-29_15:52:35/pixel_standard_deviation.npy"
        )
    )
    roi = roi.reshape(roi.shape[0], 1, roi.shape[1], roi.shape[2])
    one_roi = roi[selected_neuron_idx, :, :].reshape(1, 1, 36, 64)
    print(f"roi: {one_roi.shape}")

    # tensor (neurons, channel, height, width)
    mask = torch.from_numpy(
        load("/home/leon/csng_invariances/models/masks/2021-11-30_15:19:09/mask.npy")
    )
    mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])
    print(mask.shape)
    one_mask = mask[selected_neuron_idx, :, :].reshape(1, 1, 36, 64)
    print(f"mask: {one_mask.shape}")

    # fake epoch data: a tensor of (epochs, channel, height, width)
    # TODO save epochs data as expected
    epochs = torch.randint(low=0, high=255, size=(20, 1, 36, 64))
    print(f"epochs: {epochs[0].shape}")

    # fake cluster lists: a list of num_clusters image tensors (num_images_in_cluster, channel, height, width)
    # TODO pass cluster lists ?
    image_clusters = [
        torch.randint(low=0, high=255, size=(17, 1, 36, 64))
        for _ in range(num_clusters)
    ]

    # fake activation list: a list of num_clusters activation tensors (num_images_in_cluster, neurons)
    activation_clusters = [torch.randn(size=(17, 1)) for _ in range(num_clusters)]
    print(f"image_cluster: {image_clusters[0].shape}")
    print(f"activation_cluster: {activation_clusters[0].shape}")

    #%%
    a = torch.from_numpy(
        load(
            "/home/leon/csng_invariances/data/processed/generator/after_training/2021-12-03_12:43:36/neuron_1706/clustered/activations_cluster_0.npy"
        )
    )
    print(a.shape)
    b = torch.from_numpy(
        load(
            "/home/leon/csng_invariances/data/processed/generator/after_training/2021-12-03_12:43:36/neuron_1706/generated_images.npy"
        )
    )
    print(b.shape)
    # %%
    one_lrf = scale_tensor_to_0_1(
        lrf[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
    )
    print(f"one lrf: {one_lrf.shape}")
    one_mei = scale_tensor_to_0_1(meis[selected_neuron_idx].detach().cpu().squeeze())
    print(f"one mei: {one_mei.shape}")
    one_mei_activation = mei_activations[selected_neuron_idx].item()
    print(f"one mei acti: {one_mei_activation}")
    one_roi = scale_tensor_to_0_1(
        roi[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
    )
    print(f"one roi: {one_roi.shape}")
    one_mask = scale_tensor_to_0_1(
        mask[selected_neuron_idx, :, :, :].detach().cpu().squeeze()
    )
    print(f"one mask: {one_mask.shape}")
    epoch_slicer = [i * num_clusters % epochs.shape[0] for i in range(num_clusters)]
    epoch_tensor_list = [
        scale_tensor_to_0_1(epochs[i, :, :, :].detach().cpu().squeeze())
        for i in epoch_slicer
    ]
    print(f"epochs len: {len(epoch_tensor_list)}")
    print(f"one epoch: {epoch_tensor_list[0].shape}")
    clusters_tensor_list = [
        scale_tensor_to_0_1(i[0, :, :, :].detach().cpu().squeeze())
        for i in image_clusters
    ]
    print(f"clusters len: {len(clusters_tensor_list)}")
    print(f"one cluster image: {clusters_tensor_list[0].shape}")
    clusters_activation_list = [i[0, :].item() for i in activation_clusters]
    print(f"one cluster activation: {clusters_activation_list[0]}")
    #%%
    encoding_model = load_encoding_model(
        "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03"
    )
    #%%
    score = load(
        "/home/leon/csng_invariances/reports/scores/2021-12-02_15:46:31/score.npy"
    )
    print(score[selected_neuron_idx])
    #%%
    lrf_correlations = load_single_neuron_correlations_linear_filter(
        "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-30_15:15:09/Correlations.csv"
    )
    print(lrf_correlations.shape)
    #%%
    enc_correlations = load_single_neuron_correlations_encoding_model(
        "/home/leon/csng_invariances/reports/encoding/single_neuron_correlations/2021-11-30_15:15:09/single_neuron_correlations.npy"
    )
    print(enc_correlations.shape)
    #%%
    print(one_mask.unique())
    #%%
    generator_model = FullyConnectedGeneratorWithGaussianBlurring()
    #%%
    num_training_batches = 100
    num_generation_batches = 20
    batch_size = 64
    latent_space_dimension = 128
    training_data = [
        normal_latent_vector(batch_size, latent_space_dimension)
        for _ in range(num_training_batches)
    ]
    generation_data = [
        uniform_latent_vector(batch_size, latent_space_dimension)
        for _ in range(num_generation_batches)
    ]

# %%
