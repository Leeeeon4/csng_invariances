#%%
# from csng_invariances.encoding import load_encoding_model
# from csng_invariances.layers.mask import NaiveMask
# from csng_invariances.models.linear_filter import load_linear_filter
# from csng_invariances.training.mei import mei

# encoding_model = load_encoding_model(
#     "/Users/leongorissen/csng_invariances/models/encoding/2021-11-12_17:00:58"
# )

# # %%
# from rich import print
# from pandas import read_csv
# import torch

# device = torch.dedevice("cuda" if torch.cuda.is_available() else "cpu")
# csv = read_csv(
#     "/Users/leongorissen/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-10-29_10:31:45/Correlations.csv"
# )
# data = [float(csv.columns[1])]
# for i in csv.iloc(axis=1)[1].to_list():
#     data.append(i)

# data_tensor = torch.Tensor(data, device=device)
# # %%
# print(f"LF correlations shape: {data_tensor.shape}")
# # works as loading function
# # %%
# from csng_invariances.encoding import load_encoding_model
# from csng_invariances.data._data_helpers import load_configs, adapt_config_to_machine
# from csng_invariances.data.datasets import Lurz2021Dataset
# from csng_invariances.encoding import get_single_neuron_correlation as dnn_corrs

# model_directory = (
#     "/Users/leongorissen/csng_invariances/models/encoding/2021-11-12_17:00:58"
# )
# encoding_model = load_encoding_model(model_directory)
# print(f"Encoding model:\n{encoding_model}\n\n")
# configs = load_configs(model_directory)
# configs = adapt_config_to_machine(configs)
# #%%
# ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
# images, responses = ds.get_dataset()
# print(f"Images shape: {images.shape}")
# print(f"Response shape: {responses.shape}")
# dnn_single_neuron_correlations = dnn_corrs(
#     encoding_model,
#     images,
#     responses,
#     batch_size=configs["dataset_config"]["batch_size"],
# )
# # %%
# print(f"DNN ENC shape: {dnn_single_neuron_correlations.shape}")

# # %%
# from csng_invariances.layers.loss_function import SelectedNeuronActivation

# # %%
# from csng_invariances.data.datasets import (
#     gaussian_white_noise_image,
#     normal_latent_vector,
#     uniform_latent_vector,
# )

# # %%

# criterion = SelectedNeuronActivation()
# gwni = gaussian_white_noise_image(size=(1, 1, 36, 64))
# # %%
# from csng_invariances.metrics_statistics.select_neurons import (
#     load_score,
#     load_selected_neurons_idxs,
#     score,
#     select_neurons,
# )
# from csng_invariances.training.mei import mei

# selection_score = score(dnn_single_neuron_correlations, data_tensor)
# select_neuron_indicies = select_neurons(selection_score, 5)
# # %%
# meis = mei(criterion, encoding_model, gwni, select_neuron_indicies, lr=0.1)
# # %%
# print(meis)
# # %%
# import matplotlib.pyplot as plt

# for key, value in meis.items():
#     fig, ax = plt.subplots(figsize=(3.2, 1.8))
#     image = value.detach().numpy().squeeze()
#     image += abs(image.min())
#     image /= image.max()
#     im = ax.imshow(image, cmap="gray")
#     ax.set_title(f"MEI of neuron {key}")
#     plt.colorbar(im)
#     plt.tight_layout()
#     plt.show()
#     plt.pause(2)
#     plt.close()
# # %%
# import torch

# # %%
# a = torch.randint(0, 5225, size=(1, 1))
# # %%
# print(a.item())
# %%
########## clustering as proposed by kovacs #############
# import torch
# from sklearn.cluster import AgglomerativeClustering


# def cluster_generated_images(
#     generated_images: torch.Tensor,
#     generated_activations: torch.Tensor,
#     num_representative_samples: int = 6,
# ):
#     clusters = AgglomerativeClustering(
#             n_clusters=num_representative_samples,
#             affinity='cosine',
#             linkage='complete'
#             ).fit(generated_images)

#     clustered_images =


# def choose_representant(num_representative_samples, net, neuron, stimuli, activation_lowerbound=-np.inf, activation_upperbound=np.inf):
#     """
#      Optimization of weights in variable layer to maximize neurons output
#         Parameters:
#             num_representative_samples (int): Number of stimuli to chose
#             net (NDN): A trained neural network with variable layer
#             neuron (int): neuron id
#             stimuli (np.array): Array of generated stimuli from which to choose
#             data_filters (np.array): data_filters will be passed to NDN.train
#             activation_lowerbound (float):
#             activation_upperbound (float):
#         Returns:
#             chosen_stimuli (np.array):
#     """
#     # Filter stimuli
#     activations = net.generate_prediction(stimuli)[:, neuron]
#     stimuli = stimuli[
#         (activations <= activation_upperbound) & (
#             activations >= activation_lowerbound)
#     ]
#     activations = activations[(activations <= activation_upperbound) & (
#         activations >= activation_lowerbound)]

#     # Cluster stimuli
#     try:
#         kmeans = AgglomerativeClustering(
#             n_clusters=num_representative_samples, affinity='cosine', linkage='complete').fit(stimuli)
#         images = []
#         acts = []
#         for i in range(num_representative_samples):
#             images.append(stimuli[kmeans.labels_ == i][0, :])
#             acts.append(activations[kmeans.labels_ == i][0])
#         images = np.vstack(images)
#         acts = np.array(acts)
#     except ValueError:
#         # If clustering failed choose first `num_representative_samples` samples
#         print(
#             f'Clustering failed ... taking first {num_representative_samples} images')
#         images = stimuli[:num_representative_samples, :]
#         acts = activations[:num_representative_samples]

#     return images, acts


# %%
##############################generator.py##############################
from datetime import timedelta
from rich.progress import track
from rich import print
from numpy import save
from pathlib import Path
import torch
from csng_invariances.layers.mask import NaiveMask
from csng_invariances.mei import load_meis
from csng_invariances.metrics_statistics.correlations import (
    load_single_neuron_correlations_encoding_model,
    load_single_neuron_correlations_linear_filter,
)
from csng_invariances.models.encoding import load_encoding_model
from csng_invariances.metrics_statistics.select_neurons import (
    load_selected_neurons_idxs,
    load_score,
)
from csng_invariances.models.linear_filter import load_linear_filter
from csng_invariances.data.datasets import normal_latent_vector, uniform_latent_vector
from csng_invariances.models.generator import (
    FullyConnectedGeneratorWithGaussianBlurring,
    GrowingLinearGenerator,
    FullyConnectedGenerator,
)
from csng_invariances.training.generator import NaiveTrainer
from csng_invariances.data.preprocessing import (
    image_preprocessing,
    response_preprocessing,
)
from csng_invariances._utils.utlis import string_time
from csng_invariances.metrics_statistics.clustering import cluster_generated_images
from plotting import plot_examples_of_generated_images, plot_neuron_x_with_8_clusters

#%%


# %%
########################User setup###################################
batch_size = 64
latent_space_dimension = 128
num_training_batches = 16  # 15625
num_generation_batches = 4  # 1563
image_shape = (batch_size, 1, 36, 64)
masked = False  # Applies mask during training and generation
generate = True  # generates images after training
show_image = True  # save first image of each batch after training as *.png
selected_neuron_idxs_file = "/home/leon/csng_invariances/reports/neuron_selection/2021-11-30_15:18:09/selected_neuron_idxs.npy"
encoding_model_directory = (
    "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03"
)
bin_mask_file = "/home/leon/csng_invariances/models/masks/2021-11-30_15:19:09/mask.npy"
roi_file = "/home/leon/csng_invariances/data/processed/roi/2021-11-29_15:52:35/pixel_standard_deviation.npy"
linear_filter_file = "/home/leon/csng_invariances/data/processed/linear_filter/2021-11-30_15:15:09/evaluated_filter.npy"
meis_directory = "/home/leon/csng_invariances/data/processed/MEIs/2021-12-02_15:46:31"
score_file = "/home/leon/csng_invariances/reports/scores/2021-12-02_15:46:31/score.npy"
lrf_correlation_file = "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-30_15:15:09/Correlations.csv"
enc_correlation_file = "/home/leon/csng_invariances/reports/encoding/single_neuron_correlations/2021-12-02_15:46:31/single_neuron_correlations.npy"
# %%
###########################load model, mask and selected neurons###############
selected_neuron_idxs = load_selected_neurons_idxs(selected_neuron_idxs_file)
encoding_model = load_encoding_model(encoding_model_directory)
bin_mask = NaiveMask.load_binary_mask(bin_mask_file)
lrf = load_linear_filter(linear_filter_file)
meis = load_meis(meis_directory)
roi = NaiveMask.load_pixel_standard_deviations(roi_file)
roi = roi.reshape(roi.shape[0], 1, roi.shape[1], roi.shape[2])
score = load_score(score_file)
lrf_correlations = load_single_neuron_correlations_linear_filter(lrf_correlation_file)
enc_correlations = load_single_neuron_correlations_encoding_model(enc_correlation_file)
#%%
##########################generate training data###############################
data = [
    normal_latent_vector(batch_size, latent_space_dimension)
    for _ in range(num_training_batches)
]
eval_samples = [
    uniform_latent_vector(batch_size, latent_space_dimension)
    for _ in range(num_generation_batches)
]

print(
    f"Sum of difference of two normal vector batches: {(data[0]-data[1]).sum()}\n"
    f"Sum of difference of two uniform vector batches: "
    f"{(eval_samples[0]-eval_samples[1]).sum()}"
)
# %%
#######################RUN EXPERIMENT#############################################
from time import perf_counter

start = perf_counter()
print(f"Masking is {masked}.")
if masked:
    m = "masked"
    mask = bin_mask
else:
    m = "not_masked"
    mask = None
t = string_time()
config = {}
config["Timestamp"] = t
intermediates = {}
for neuron_counter, neuron in enumerate(selected_neuron_idxs):
    print(f"Neuron {neuron_counter+1} / {len(selected_neuron_idxs)}")
    generator_model = FullyConnectedGeneratorWithGaussianBlurring(
        output_shape=image_shape,
        latent_space_dimension=latent_space_dimension,
        sigma=0.8,
        batch_norm=True,
    )
    # sigma 0.5 still had artifact, sigma 1 not
    generator_trainer = NaiveTrainer(
        generator_model=generator_model,
        encoding_model=encoding_model,
        data=data,
        mask=mask,
        image_preprocessing=image_preprocessing,
        response_preprocessing=response_preprocessing,
        epochs=20,
        weight_decay=0.1,
        show_development=True,
        prep_video=False,
        config=config,
    )
    generator_model, epochs_images, config = generator_trainer.train(neuron)
    t = config["Timestamp"] + "_" + config["wandb_name"]
    if generate:
        data_directory = (
            Path.cwd()
            / "data"
            / "processed"
            / "generator"
            / "after_training"
            / t
            / f"neuron_{neuron}"
        )
        data_directory.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():  # gradient not needed, increases speed
            generated_images = torch.empty(
                size=(
                    batch_size * num_training_batches,
                    image_shape[1],
                    image_shape[2],
                    image_shape[3],
                ),
                device="cuda",
            )
            tensors = []
            for batch_counter, eval_sample in track(
                enumerate(eval_samples),
                total=len(eval_samples),
                description=f"Generating images: ",
            ):

                # eval_sample is tensor of shape (batch_size, latent_vector_dimension)
                sample_image = generator_model(eval_sample)

                # sample_image is tensor of shape (batch_size, 1, height, width)
                if masked:
                    masking = NaiveMask(mask, neuron)
                    masked_image = masking(sample_image)
                else:
                    masked_image = sample_image

                # preprocessed_image is N(0,1) normalized and scaled to be [0,1]
                preprocessed_image = image_preprocessing(masked_image)

                # plot examples if True
                if show_image:
                    image_directory = (
                        Path.cwd()
                        / "reports"
                        / "figures"
                        / "generator"
                        / "after_training"
                        / t
                        / f"neuron_{neuron}"
                    )
                    image_directory.mkdir(parents=True, exist_ok=True)
                    activation = encoding_model(preprocessed_image)
                    plot_examples_of_generated_images(
                        selected_neuron_idx=neuron,
                        batch_counter=batch_counter,
                        sample_image=sample_image,
                        masked_image=masked_image,
                        preprocessed_image=preprocessed_image,
                        image_directory=image_directory,
                        encoding_model=encoding_model,
                    )

                # add into one vector
                tensors.append(preprocessed_image)
            generated_images = torch.cat(tensors, dim=0)

            # compute activations and save data
            activations = encoding_model(generated_images)
            activations = activations[:, neuron]
            file_name = f"generated_images.npy"
            save(
                file=data_directory / file_name,
                arr=generated_images.detach().cpu().numpy(),
            )
            file_name = f"activations.npy"
            save(
                file=data_directory / file_name,
                arr=activations.detach().cpu().numpy(),
            )

            # cluster images (and activations accordingly) to detect different images
            clustered_images, clustered_activations = cluster_generated_images(
                generated_images, activations, neuron, show=False
            )

            # save clusters
            cluster_directory = data_directory / "clustered"
            cluster_directory.mkdir(parents=True, exist_ok=True)
            for cluster_counter, (image_cluster, activation_cluster) in enumerate(
                zip(clustered_images, clustered_activations)
            ):
                save(
                    file=cluster_directory / f"images_cluster_{cluster_counter}.npy",
                    arr=image_cluster.detach().cpu().numpy(),
                )
                save(
                    file=cluster_directory
                    / f"activations_cluster_{cluster_counter}.npy",
                    arr=activation_cluster.detach().cpu().numpy(),
                )

            # print(clustered_activations)
            plot_neuron_x_with_8_clusters(
                selected_neuron_idx=neuron,
                lrf=lrf,
                meis=meis,
                roi=roi,
                mask=mask,
                epochs_images=epochs_images,
                clustered_images=clustered_images,
                clustered_activations=clustered_activations,
                score=score,
                lrf_correlations=lrf_correlations,
                enc_correlations=enc_correlations,
                training_data=data,
                generation_data=eval_samples,
                encoding_model=encoding_model,
                generator_model=generator_model,
                config=config,
            )

    intermediate = perf_counter()
    intermediates[neuron_counter] = intermediate
    if neuron_counter == 0:

        print(f"Current neuron {neuron} took: {(intermediate-start):.2f}s")
    else:
        print(
            f"Current neuron {neuron} took: {(intermediates[neuron_counter]-intermediates[neuron_counter-1]):.2f}s"
        )
end = perf_counter()
print(f"Complete process took {timedelta(seconds=(end-start))}")


# %%
