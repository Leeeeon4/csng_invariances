"""Handels dataset presented in Antolik et al. 2016.
"""


import numpy as np
import torch
import requests
import zipfile

from torch.utils.data import Dataset
from rich import print
from rich.progress import track
from pathlib import Path

from csng_invariances.data._data_helpers import make_directories
from warnings import warn

warn(
    f"The module {__name__} is deprecated. Please use csng_invariances.data.datasets instead.",
    DeprecationWarning,
    stacklevel=2,
)


def get_antolik2016_data():
    """
    Download and unzip Antolik et al. 2016 dataset

    Returns:
        pathlib.PurePath: Path to unzipped data.
    """

    print("Downloading dataset")
    url = "https://doi.org/10.1371/journal.pcbi.1004927.s001"
    r = requests.get(url)
    zip = "data.zip"
    with open(zip, "wb") as code:
        code.write(r.content)
    directories = make_directories()
    data_external_antolik_directory = directories[1] / "antolik2016"
    data_external_antolik_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip, "r") as zip_ref:
        zip_ref.extractall(data_external_antolik_directory)
    zip = Path.cwd() / zip
    zip.unlink()
    print("Finished downlading and extracting dataset")
    return data_external_antolik_directory / "Data"


def get_dataloaders():
    """
    Generate dictionary of regions and dictionaries of dataset types and dataset
        objects.

    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """
    # TODO redo, so that dataloaders is as expected for Lurz model
    dataloaders = {
        "train": {
            "region1": Antolik2016Dataset("region1", "training"),
            "region2": Antolik2016Dataset("region2", "training"),
            "region3": Antolik2016Dataset("region3", "training"),
        },
        "validation": {
            "region1": Antolik2016Dataset("region1", "validation"),
            "region2": Antolik2016Dataset("region2", "validation"),
            "region3": Antolik2016Dataset("region3", "validation"),
        },
    }
    return dataloaders


def get_complete_dataset(dataloaders, key="train", region="region1"):
    """
    Load complete dataset to memory.

    Args:
        dataloaders (OrderedDict): dictionary of dictionaries where the first level
            keys are 'train', 'validation', and 'test', and second level keys are
            regions.
        key (string, optional): Data. Defaults to train.
        region (strin, optional): Region. Defaults to 'region1'.

    Returns:
        Tuple: Tuple of images_tensor and responses_tensor
    """

    dataset = dataloaders[key][region]
    images_tensor = torch.empty(
        [
            len(dataset),
            dataset[0][0].shape[0],
            dataset[0][0].shape[1],
            dataset[0][0].shape[2],
        ]
    )
    responses_tensor = torch.empty([len(dataset), dataset[0][1].shape[0]])
    for i in range(len(dataset)):
        image = dataset[i][0]
        response = dataset[i][1]
        images_tensor[i] = image
        responses_tensor[i] = response

    print(
        f"The {key} set of dataset {region} contains the responses of "
        f"{dataset[0][1].shape[0]} neurons to {len(dataset)} images."
    )

    return images_tensor, responses_tensor


def get_images_reponses(region="region1"):
    """Return dataset of region where train and validation is concatonated.

    Args:
        region (str, optional): Region. Defaults to 'region1'.

    Returns:
        tuple: tuple of image and response tensor.
    """
    images, responses = get_complete_dataset(get_dataloaders(), region=region)
    v_images, v_responses = get_complete_dataset(
        get_dataloaders(), key="validation", region=region
    )
    torch.cat([images, v_images], dim=0)
    torch.cat([responses, v_responses], dim=0)
    return images, responses


# map-style datasets


class Antolik2016Dataset(Dataset):
    """
    Create custom map-style torch.nn.utils.data.Dataset.

    Map-style mouse V1 dataset of stimulus (images) and neural response
    (average number of spikes) as prensented in Antolik et al. 2016:
    Model Constrained by Visual Hierachy Improves Prediction of
    Neural Responses to Natural Scenes.
    Contains data for three regions in two animals (mice):\n
    training_images: n images (31x31) as npy file\n
    training_set: m neural responses as npy file\n
    validation_images: 50 images (31x31) as npy file\n
    validation_set: m neural responses as npy file
    """

    def __init__(self, region, dataset_type="training", transform=None):
        """Instantiate class.

        Args:
            data_dir (pathlib Path): Path to data directory
            region (string): region to examine
                options: 'region1', 'region2', 'region3'
            dataset_type (string, default = 'training'): dataset type
                options: 'training', 'validation'
            transform (callable, optional): Optional transforms
                to be applied to sample
        """

        assertation_message = "dataset_type should be one of 'training', 'validation'"
        assert dataset_type in set(["training", "validation"]), assertation_message
        assertation_message = "region should be one of'region1', 'region2', 'region3'"
        assert region in set(["region1", "region2", "region3"]), assertation_message
        self.data_dir = Path.cwd() / "data" / "external" / "antolik2016" / "Data"
        if (self.data_dir / "README").exists() is False:
            make_directories()
            self.data_dir = get_antolik2016_data()
        self.region = region

        self.training_responses = np.load(
            self.data_dir / self.region / "training_set.npy"
        )
        # response shape: image_count, neuron_count
        self.training_images = np.load(
            self.data_dir / self.region / "training_inputs.npy"
        )
        self.training_count = self.training_images.shape[0]
        self.channels = 1
        self.dim1 = 31
        self.dim2 = 31
        # reshape to image representation (n, c, h, w)
        self.training_images = self.training_images.reshape(
            self.training_count, self.channels, self.dim1, self.dim2
        )

        self.validation_responses = np.load(
            self.data_dir / self.region / "validation_set.npy"
        )
        self.validation_images = np.load(
            self.data_dir / self.region / "validation_inputs.npy"
        )
        self.validation_count = self.validation_images.shape[0]
        # reshape to image representation (n, h, w, c)
        self.validation_images = self.validation_images.reshape(
            self.validation_count, self.channels, self.dim1, self.dim2
        )

        self.dataset_type = dataset_type
        self.transform = transform

    def __len__(self):
        if self.dataset_type == "training":
            length = self.training_responses.shape[0]
        else:
            length = 50  # validation set size always 50
        return length

    def __getitem__(self, idx):
        """
        Enable indexing of Dataset object.

        Args:
            idx (int): index of item to get.

        Returns:
            (touple): touple of image and label tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataset_type == "training":
            image = torch.tensor(
                self.training_images[idx, :, :].reshape(
                    self.channels, self.dim1, self.dim2
                )
            )
            label = torch.tensor(self.training_responses[idx, :])
        else:
            image = torch.tensor(
                self.validation_images[idx, :, :].reshape(
                    self.channels, self.dim1, self.dim2
                )
            )
            label = torch.tensor(self.validation_responses[idx, :])
        if self.transform:
            image = self.transform(image)
        sample = (image, label)
        return sample

    def get_neuroncount(self):
        """
        Returns:
            neuroncount (int): number of neurons observed in the specific dataset
        """

        if self.dataset_type == "training":
            neuroncount = self.training_responses.shape[1]
        else:
            neuroncount = self.validation_responses.shape[1]
        return neuroncount


if __name__ == "__main__":
    pass
