#%%
from csng_invariances.encoding import load_encoding_model
from csng_invariances.training.mei import mei

encoding_model = load_encoding_model(
    "/Users/leongorissen/csng_invariances/models/encoding/2021-11-12_17:00:58"
)

# %%
from rich import print
from pandas import read_csv
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
csv = read_csv(
    "/Users/leongorissen/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-10-29_10:31:45/Correlations.csv"
)
data = [float(csv.columns[1])]
for i in csv.iloc(axis=1)[1].to_list():
    data.append(i)

data_tensor = torch.Tensor(data, device=device)
# %%
print(f"LF correlations shape: {data_tensor.shape}")
# works as loading function
# %%
from csng_invariances.encoding import load_encoding_model
from csng_invariances.data._data_helpers import load_configs, adapt_config_to_machine
from csng_invariances.data.datasets import Lurz2021Dataset
from csng_invariances.encoding import get_single_neuron_correlation as dnn_corrs

model_directory = (
    "/Users/leongorissen/csng_invariances/models/encoding/2021-11-12_17:00:58"
)
encoding_model = load_encoding_model(model_directory)
print(f"Encoding model:\n{encoding_model}\n\n")
configs = load_configs(model_directory)
configs = adapt_config_to_machine(configs)
#%%
ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
images, responses = ds.get_dataset()
print(f"Images shape: {images.shape}")
print(f"Response shape: {responses.shape}")
dnn_single_neuron_correlations = dnn_corrs(
    encoding_model,
    images,
    responses,
    batch_size=configs["dataset_config"]["batch_size"],
)
# %%
print(f"DNN ENC shape: {dnn_single_neuron_correlations.shape}")

# %%
from csng_invariances.layers.loss_function import SelectedNeuronActivation

# %%
from csng_invariances.data.datasets import (
    gaussian_white_noise_image,
    normal_latent_vector,
    uniform_latent_vector,
)

# %%

criterion = SelectedNeuronActivation()
gwni = gaussian_white_noise_image(size=(1, 1, 36, 64))
# %%
from csng_invariances.metrics_statistics.select_neurons import score, select_neurons
from csng_invariances.training.mei import mei

selection_score = score(dnn_single_neuron_correlations, data_tensor)
select_neuron_indicies = select_neurons(selection_score, 5)
# %%
meis = mei(criterion, encoding_model, gwni, select_neuron_indicies, lr=0.1)
# %%
print(meis)
# %%
import matplotlib.pyplot as plt

for key, value in meis.items():
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    image = value.detach().numpy().squeeze()
    image += abs(image.min())
    image /= image.max()
    im = ax.imshow(image, cmap="gray")
    ax.set_title(f"MEI of neuron {key}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
    plt.pause(2)
    plt.close()
# %%
