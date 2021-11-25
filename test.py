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
print(data_tensor)
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
configs = load_configs(model_directory)
configs = adapt_config_to_machine(configs)
ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
images, responses = ds.get_dataset()
dnn_single_neuron_correlations = dnn_corrs(
    encoding_model,
    images,
    responses,
    batch_size=configs["dataset_config"]["batch_size"],
)
print(dnn_single_neuron_correlations)

# %%
print(dnn_single_neuron_correlations.shape)
# %%
print(data_tensor.shape)
# %%
from csng_invariances.losses.loss_modules import SelectedNeuronActivation

# %%
def gaussian_white_noise_image(size: Tuple[int], device: str = None) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gwni = torch.randint(
        low=0,
        high=255,
        size=size,
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )
    return gwni


class LatentVector:
    def __init__(
        self,
        num_vectors: int = 1_000_000,
        latent_space_dimension: int = 128,
        device: str = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector = torch.empty(
            size=(num_vectors, latent_space_dimension),
            device=device,
            dtype=torch.float,
            requires_grad=True,
        )


def normal_latent_vector(
    num_vectors: int = 1_000_000,
    latent_space_dimension: int = 128,
    device: str = None,
    mean: float = 0,
    std: float = 1,
) -> torch.Tensor:
    """Returns a latent tensor with gradient drawn form normal distribution.

    Args:
        num_vectors (int, optional): Number of Vectors. Defaults to 1_000_000.
        latent_space_dimension (int, optional): Dimensionality of Vectors. Defaults to 128.
        device (str, optional): Device, if none tries to default to cuda. Defaults to None.
        mean (float, optional): mean of normal distribution to sample from. Defaults to 0.
        std (float, optional): std of normal distribution to sample from. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of size (num_vectors, latent_space_dimension) sampled
            from normal distribution with mean: mean and std: std.
    """
    tensor = LatentVector(num_vectors, latent_space_dimension, device)
    return torch.nn.init.normal_(tensor.vector, mean, std)


def uniform_latent_vector(
    num_vectors: int = 1_000_000,
    latent_space_dimension: int = 128,
    device: str = None,
    low: float = -2,
    high: float = 2,
) -> torch.Tensor:
    """Returns a latent tensor with gradient drawn form uniform distribution.

    Args:
        num_vectors (int, optional): Number of Vectors. Defaults to 1_000_000.
        latent_space_dimension (int, optional): Dimensionality of Vectors. Defaults to 128.
        device (str, optional): Device, if none tries to default to cuda. Defaults to None.
        low (float, optional): low of uniform distribution to sample from. Defaults to 0.
        high (float, optional): high of uniform distribution to sample from. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of size (num_vectors, latent_space_dimension) sampled
            from normal distribution with mean: mean and std: std.
    """
    tensor = LatentVector(num_vectors, latent_space_dimension, device)
    return torch.nn.init.uniform_(tensor.vector, low, high)


# %%


criterion = SelectedNeuronActivation()
gwni = gaussian_white_noise_image(size=(1, 1, 36, 64))
# %%
from csng_invariances.select_neurons import score, select_neurons
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
