"""Module with different dataset classes."""

import requests
import torch
import numpy as np

from torch.utils.data import Dataset
from rich import print
from zipfile import ZipFile
from pathlib import Path
from typing import Callable, Tuple
from collections import OrderedDict
from itertools import zip_longest
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from nnfabrik.utility.nn_helpers import set_random_seed

# local fork of neuralpredictors
from csng_invariances._neuralpredictors.data.datasets import (
    StaticImageSet,
    FileTreeDataset,
)
from csng_invariances._neuralpredictors.data.transforms import (
    Subsample,
    ToTensor,
    NeuroNormalizer,
    AddBehaviorAsChannels,
    SelectInputChannel,
)
from csng_invariances._neuralpredictors.data.samplers import SubsetSequentialSampler


from csng_invariances.data._data_helpers import get_oracle_dataloader


class NeuralDataset(Dataset):
    def __init__(
        self,
        device: str = None,
        image_preprocessing: Callable = None,
        response_preprocessing: Callable = None,
    ) -> None:
        super().__init__()
        self.device = self._parse_device(device)
        self.image_preprocessing = self._parse_function(image_preprocessing)
        self.response_preprocessing = self._parse_function(response_preprocessing)

    def download(self):
        print(f"Downloading dataset from {self.URL}")
        r = requests.get(self.URL)
        zip_file = "data.zip"
        with open(zip_file, "wb") as f:
            f.write(r.content)
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(self.data_path)
        zip_file = Path.cwd() / zip_file
        zip_file.unlink()
        print("Finished downloading and extracting.")

    def _parse_function(self, func: Callable) -> Callable:
        if func is None:

            def no_processing(x: torch.tensor) -> torch.Tensor:
                return x

            return no_processing
        return func

    def _parse_device(self, device: str) -> str:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        return self.device


class Antolik2016Dataset(NeuralDataset):
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

    def __init__(
        self,
        device: str = None,
        region: str = "region1",
        dataset_type: str = None,
        image_preprocessing: Callable = None,
        response_preprocessing: Callable = None,
    ):
        """Instantiate Dataset

        Args:
            device (str, optional): Computation device. Defaults to None.
            region (str, optional): Dataset region, options are 'region1',
                'region2' and 'region3'. Defaults to "region1".
            dataset_type (str, optional): Dataset type, options are 'training'
                and 'validation'. Defaults to "training".
            image_preprocessing (Callable, optional): Image-preprocessing
                function. Defaults to None.
            response_preprocessing (Callable, optional): Response-preprocessing
                function. Defaults to None.
        """
        super().__init__(
            device=device,
            image_preprocessing=image_preprocessing,
            response_preprocessing=response_preprocessing,
        )
        self.data_path = Path.cwd() / "data" / "external" / "antolik2016"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.URL = "https://doi.org/10.1371/journal.pcbi.1004927.s001"
        self.dataset_type = dataset_type
        self.region = region

        # set dataset type:
        if dataset_type is None:
            self.dataset_type = "training"
        else:
            self.dataset_type = dataset_type

        # download if not yet downloaded
        if (self.data_path / "Data" / "README").exists() is False:
            super().download()
        self.data_path = self.data_path / "Data"

        # load responses
        self.responses = torch.from_numpy(
            np.load(self.data_path / self.region / f"{self.dataset_type}_set.npy")
        ).to(self.device)
        self.neuron_count = self.responses.shape[1]
        self.val_responses = torch.from_numpy(
            np.load(self.data_path / self.region / f"validation_set.npy")
        ).to(self.device)

        # load images
        self.images = torch.from_numpy(
            np.load(self.data_path / self.region / f"{self.dataset_type}_inputs.npy")
        ).to(self.device)
        self.val_images = torch.from_numpy(
            np.load(self.data_path / self.region / "validation_inputs.npy")
        ).to(self.device)
        self.image_count = self.images.shape[0]
        self.channels = 1
        self.height = 31
        self.width = 31
        # reshape to image representation (n, c, h, w)
        self.images = self.images.reshape(
            self.image_count, self.channels, self.height, self.width
        )
        self.val_images = self.val_images.reshape(
            50, self.channels, self.height, self.width
        )

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx: int) -> Tuple:
        """
        Enable indexing of Dataset object.

        Args:
            idx (Int): index of item to get.

        Returns:
            (Tuple): tuple of image and response tensor.
        """
        image = self.image_preprocessing(self.images[idx, :, :, :])
        response = self.response_preprocessing(self.responses[idx, :])
        return (image, response)

    def get_dataset(self, concat: bool = False) -> Tuple:
        """Load and preprocess complete dataset.

        Returns:
            Tuple: Tuple of preprocessed images and reponses tensor
        """
        images = self.image_preprocessing(self.images)
        responses = self.response_preprocessing(self.responses)

        if concat:
            val_images = self.image_preprocessing(self.val_images)
            images = torch.cat([images, val_images], dim=0)
            val_responses = self.response_preprocessing(self.val_responses)
            responses = torch.cat([responses, val_responses], dim=0)

        print(
            f"The {('concatenated' if concat else self.dataset_type)} set of dataset "
            f"{self.region} contains the responses of {responses.shape[1]} neurons "
            f"to {images.shape[0]} images."
        )
        return (images, responses)

    def get_dataloaders(self):
        # TODO implement get_dataloaders function for Antolik dataset.
        None


class Lurz2021Dataset(NeuralDataset):
    def __init__(
        self,
        dataset_config: dict,
        device: str = None,
        dataset_type: str = "train",
        dataset_name: str = "20457-5-9-0",
        image_preprocessing: Callable = None,
        response_preprocessing: Callable = None,
    ) -> None:
        super().__init__(
            device=device,
            image_preprocessing=image_preprocessing,
            response_preprocessing=response_preprocessing,
        )
        self.URL = "https://gin.g-node.org/cajal/Lurz2020/archive/master.zip"
        self.data_path = Path.cwd() / "data" / "external" / "lurz2020"
        if (self.data_path / "README.md").is_file() is False:
            super().download()
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataloaders = Lurz2021Dataset.static_loaders(**self.dataset_config)

    def get_dataloaders(self):
        return self.dataloaders

    def get_dataset(
        self, dataset_type: str = None, concat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load dataset to memory.

        Args:
            dataset_type (str, optional): Dataset type, options are 'train',
                'validation', 'test'. If None, 'train' is used. Defaults to None.
            concat (bool, optional): If True, all dataset_types are concatenated.
                Defaults to False.

        Returns:
            Tuple: Tuple of image and response Tensor.
        """
        print("Loading Dataset to memory.")
        if concat:
            images, responses = [], []
            for typ in ["train", "validation", "test"]:
                for x, y in self.get_dataloaders()[typ][self.dataset_name]:
                    images.append(x.squeeze().cpu().data.numpy())
                    responses.append(y.squeeze().cpu().data.numpy())
        elif dataset_type is None:
            images, responses = [], []
            for x, y in self.get_dataloaders()[self.dataset_type][self.dataset_name]:
                images.append(x.squeeze().cpu().data.numpy())
                responses.append(y.squeeze().cpu().data.numpy())
        else:
            images, responses = [], []
            for x, y in self.get_dataloaders()[dataset_type][self.dataset_name]:
                images.append(x.squeeze().cpu().data.numpy())
                responses.append(y.squeeze().cpu().data.numpy())

        images_tensor = torch.tensor(np.vstack(images), device=self.device)
        responses_tensor = torch.tensor(np.vstack(responses), device=self.device)
        self.image_count, self.height, self.width = images_tensor.shape
        self.channels = 1
        self.neuron_count = responses_tensor.shape[1]

        self.responses = responses_tensor
        self.images = images_tensor.reshape(
            self.image_count, self.channels, self.height, self.width
        )

        print(
            f"The {'concatonated' if concat else self.dataset_type} set of dataset {self.dataset_name} contains the responses of {self.neuron_count} neurons to {self.image_count} images."
        )
        return self.images, self.responses

    @staticmethod
    def static_loader(
        path,
        batch_size,
        areas=None,
        layers=None,
        tier=None,
        neuron_ids=None,
        neuron_n=None,
        exclude_neuron_n=0,
        neuron_base_seed=None,
        image_ids=None,
        image_n=None,
        image_base_seed=None,
        get_key=False,
        cuda=True,
        normalize=True,
        exclude="images",
        include_behavior=False,
        select_input_channel=None,
        file_tree=True,
        return_test_sampler=False,
        oracle_condition=None,
        image_preprocessing=None,
        response_preprocessing=None,
    ):
        """
        returns a single data loader

        Args:
            path (str): path for the dataset
            batch_size (int): batch size.
            areas (list, optional): the visual area.
            layers (list, optional): the layer from visual area.
            tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
            neuron_ids (list, optional): select neurons by their ids.
            neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
            neuron_base_seed (float, optional): base seed for neuron selection. Get's multiplied by neuron_n to obtain final seed
            exclude_neuron_n (int): the first <exclude_neuron_n> neurons will be excluded (given a neuron_base_seed),
                                    then <neuron_n> neurons will be drawn from the remaining neurons.
            image_ids (list, optional): select images by their ids.
            image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
            image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
            get_key (bool, optional): whether to return the data key, along with the dataloaders.
            cuda (bool, optional): whether to place the data on gpu or not.
            normalize (bool, optional): whether to normalize the data (see also exclude)
            exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
            include_behavior (bool, optional): whether to include behavioral data
            select_input_channel (int, optional): Only for color images. Select a color channel
            file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
            return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
            oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler
            image_preprocessing (function, optional): Function for all image preprocessing steps. Defaults to None.
            response_preprocessing (function, optional): Function for all response preprocessing stepts. Defaults to None.

        Returns:
            if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
            if get_key is True it returns the data_key (as the first output) followed by the dataloder dictionary.

        """
        assert any(
            [image_ids is None, all([image_n is None, image_base_seed is None])]
        ), "image_ids can not be set at the same time with anhy other image selection criteria"
        assert any(
            [
                neuron_ids is None,
                all(
                    [
                        neuron_n is None,
                        neuron_base_seed is None,
                        areas is None,
                        layers is None,
                        exclude_neuron_n == 0,
                    ]
                ),
            ]
        ), "neuron_ids can not be set at the same time with any other neuron selection criteria"
        assert any(
            [exclude_neuron_n == 0, neuron_base_seed is not None]
        ), "neuron_base_seed must be set when exclude_neuron_n is not 0"
        data_key = (
            path.split("static")[-1]
            .split(".")[0]
            .replace("preproc", "")
            .replace("_nobehavior", "")
        )

        if file_tree:
            dat = (
                FileTreeDataset(
                    path,
                    "images",
                    "responses",
                    "behavior",
                    image_preprocessing=image_preprocessing,
                    response_preprocessing=response_preprocessing,
                )
                if include_behavior
                else FileTreeDataset(
                    path,
                    "images",
                    "responses",
                    image_preprocessing=image_preprocessing,
                    response_preprocessing=response_preprocessing,
                )
            )
        else:
            dat = (
                StaticImageSet(path, "images", "responses", "behavior")
                if include_behavior
                else StaticImageSet(path, "images", "responses")
            )

        assert (
            include_behavior and select_input_channel
        ) is False, (
            "Selecting an Input Channel and Adding Behavior can not both be true"
        )

        # The permutation MUST be added first and the conditions below MUST NOT be based on the original order
        # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
        conds = np.ones(len(dat.neurons.area), dtype=bool)
        if areas is not None:
            conds &= np.isin(dat.neurons.area, areas)
        if layers is not None:
            conds &= np.isin(dat.neurons.layer, layers)
        idx = np.where(conds)[0]
        if neuron_n is not None:
            random_state = np.random.get_state()
            if neuron_base_seed is not None:
                np.random.seed(
                    neuron_base_seed * neuron_n
                )  # avoid nesting by making seed dependent on number of neurons
            assert (
                len(dat.neurons.unit_ids) >= exclude_neuron_n + neuron_n
            ), "After excluding {} neurons, there are not {} neurons left".format(
                exclude_neuron_n, neuron_n
            )
            neuron_ids = np.random.choice(
                dat.neurons.unit_ids, size=exclude_neuron_n + neuron_n, replace=False
            )[exclude_neuron_n:]
            np.random.set_state(random_state)
        if neuron_ids is not None:
            idx = [
                np.where(dat.neurons.unit_ids == unit_id)[0][0]
                for unit_id in neuron_ids
            ]

        more_transforms = [Subsample(idx), ToTensor(cuda)]
        if normalize:
            more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))

        if include_behavior:
            more_transforms.insert(0, AddBehaviorAsChannels())

        if select_input_channel is not None:
            more_transforms.insert(0, SelectInputChannel(select_input_channel))

        dat.transforms.extend(more_transforms)

        if return_test_sampler:
            print("Returning only test sampler with repeats...")
            dataloader = get_oracle_dataloader(
                dat, oracle_condition=oracle_condition, file_tree=file_tree
            )
            return (data_key, {"test": dataloader}) if get_key else {"test": dataloader}

        # subsample images
        dataloaders = {}
        keys = [tier] if tier else ["train", "validation", "test"]
        tier_array = dat.trial_info.tiers if file_tree else dat.tiers
        image_id_array = (
            dat.trial_info.frame_image_id if file_tree else dat.info.frame_image_id
        )
        for tier in keys:
            # sample images
            if tier == "train" and image_ids is not None:
                subset_idx = [
                    np.where(image_id_array == image_id)[0][0] for image_id in image_ids
                ]
                assert (
                    sum(tier_array[subset_idx] != "train") == 0
                ), "image_ids contain validation or test images"
            elif tier == "train" and image_n is not None:
                random_state = np.random.get_state()
                if image_base_seed is not None:
                    np.random.seed(
                        image_base_seed * image_n
                    )  # avoid nesting by making seed dependent on number of images
                subset_idx = np.random.choice(
                    np.where(tier_array == "train")[0], size=image_n, replace=False
                )
                np.random.set_state(random_state)
            else:
                subset_idx = np.where(tier_array == tier)[0]

            sampler = (
                SubsetRandomSampler(subset_idx)
                if tier == "train"
                else SubsetSequentialSampler(subset_idx)
            )
            dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

        # create the data_key for a specific data path
        return (data_key, dataloaders) if get_key else dataloaders

    @staticmethod
    def static_loaders(
        paths,
        batch_size,
        seed=None,
        areas=None,
        layers=None,
        tier=None,
        neuron_ids=None,
        neuron_n=None,
        exclude_neuron_n=0,
        neuron_base_seed=None,
        image_ids=None,
        image_n=None,
        image_base_seed=None,
        cuda=True,
        normalize=True,
        include_behavior=False,
        exclude="images",
        select_input_channel=None,
        file_tree=True,
        return_test_sampler=False,
        oracle_condition=None,
        image_preprocessing=None,
        response_preprocessing=None,
    ):
        """
        Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).

        Args:
            paths (list): list of paths for the datasets
            batch_size (int): batch size.
            seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
            areas (list, optional): the visual area.
            layers (list, optional): the layer from visual area.
            tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
            neuron_ids (list, optional): List of lists of neuron_ids. Make sure the order is the same as in paths
            neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
            exclude_neuron_n (int): the first <exclude_neuron_n> neurons will be excluded (given a neuron_base_seed),
                                    then <neuron_n> neurons will be drawn from the remaining neurons.
            neuron_base_seed (float, optional): base seed for neuron selection. Get's multiplied by neuron_n to obtain final seed
            image_ids (list, optional): List of lists of image_ids. Make sure the order is the same as in paths
            image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
            image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
            cuda (bool, optional): whether to place the data on gpu or not.
            normalize (bool, optional): whether to normalize the data (see also exclude)
            exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
            include_behavior (bool, optional): whether to include behavioral data
            select_input_channel (int, optional): Only for color images. Select a color channel
            file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
            return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
            oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler

        Returns:
            dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
        """
        set_random_seed(seed)
        dls = OrderedDict({})
        keys = [tier] if tier else ["train", "validation", "test"]
        for key in keys:
            dls[key] = OrderedDict({})

        neuron_ids = [neuron_ids] if neuron_ids is None else neuron_ids
        image_ids = [image_ids] if image_ids is None else image_ids
        for path, neuron_id, image_id in zip_longest(
            paths, neuron_ids, image_ids, fillvalue=None
        ):
            data_key, loaders = Lurz2021Dataset.static_loader(
                path,
                batch_size,
                areas=areas,
                layers=layers,
                cuda=cuda,
                tier=tier,
                get_key=True,
                neuron_ids=neuron_id,
                neuron_n=neuron_n,
                exclude_neuron_n=exclude_neuron_n,
                neuron_base_seed=neuron_base_seed,
                image_ids=image_id,
                image_n=image_n,
                image_base_seed=image_base_seed,
                normalize=normalize,
                include_behavior=include_behavior,
                exclude=exclude,
                select_input_channel=select_input_channel,
                file_tree=file_tree,
                return_test_sampler=return_test_sampler,
                oracle_condition=oracle_condition,
                image_preprocessing=image_preprocessing,
                response_preprocessing=response_preprocessing,
            )
            for k in dls:
                dls[k][data_key] = loaders[k]

        return dls

    @staticmethod
    def static_shared_loaders(
        paths,
        batch_size,
        seed=None,
        areas=None,
        layers=None,
        tier=None,
        multi_match_ids=None,
        multi_match_n=None,
        exclude_multi_match_n=0,
        multi_match_base_seed=None,
        image_ids=None,
        image_n=None,
        image_base_seed=None,
        cuda=True,
        normalize=True,
        include_behavior=False,
        exclude="images",
        select_input_channel=None,
        return_test_sampler=False,
        oracle_condition=None,
    ):
        """
        Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
        The datasets must have matched neurons. Only the file tree format is supported.

        Args:
            paths (list): list of paths for the datasets
            batch_size (int): batch size.
            seed (int): seed. Not really needed because there are neuron and image seed. But nnFabrik requires it.
            areas (list, optional): the visual area.
            layers (list, optional): the layer from visual area.
            tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
            multi_match_ids (list, optional): List of multi_match_ids according to which the respective neuron_ids are drawn for each dataset in paths
            multi_match_n (int, optional): number of neurons to select randomly. Can not be set together with multi_match_ids
            exclude_multi_match_n (int): the first <exclude_multi_match_n> matched neurons will be excluded (given a multi_match_base_seed),
                                    then <multi_match_n> matched neurons will be drawn from the remaining neurons.
            multi_match_base_seed (float, optional): base seed for neuron selection. Get's multiplied by multi_match_n to obtain final seed
            image_ids (list, optional): List of lists of image_ids. Make sure the order is the same as in paths
            image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
            image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
            cuda (bool, optional): whether to place the data on gpu or not.
            normalize (bool, optional): whether to normalize the data (see also exclude)
            exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
            include_behavior (bool, optional): whether to include behavioral data
            select_input_channel (int, optional): Only for color images. Select a color channel
            return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
            oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler
            image_preprocessing (function, optional): Function for all image preprocessing steps. Defaults to None.
            response_preprocessing (function, optional): Function for all response preprocessing stepts. Defaults to None.

        Returns:
            dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
        """
        set_random_seed(seed)
        assert (
            len(paths) != 1
        ), "Only one dataset was specified in 'paths'. When using the 'static_shared_loaders', more than one dataset has to be passed."
        assert any(
            [
                multi_match_ids is None,
                all(
                    [
                        multi_match_n is None,
                        multi_match_base_seed is None,
                        exclude_multi_match_n == 0,
                    ]
                ),
            ]
        ), "multi_match_ids can not be set at the same time with any other multi_match selection criteria"
        assert any(
            [exclude_multi_match_n == 0, multi_match_base_seed is not None]
        ), "multi_match_base_seed must be set when exclude_multi_match_n is not 0"
        # Collect overlapping multi matches
        multi_unit_ids, per_data_set_ids, given_neuron_ids = [], [], []
        match_set = None
        for path in paths:
            data_key, dataloaders = Lurz2021Dataset.static_loader(
                path=path, batch_size=100, get_key=True
            )
            dat = dataloaders["train"].dataset
            multi_unit_ids.append(dat.neurons.multi_match_id)
            per_data_set_ids.append(dat.neurons.unit_ids)
            if match_set is None:
                match_set = set(multi_unit_ids[-1])
            else:
                match_set &= set(multi_unit_ids[-1])
            if multi_match_ids is not None:
                assert set(multi_match_ids).issubset(
                    dat.neurons.multi_match_id
                ), "Dataset {} does not contain all multi_match_ids".format(path)
                neuron_idx = [
                    np.where(dat.neurons.multi_match_id == multi_match_id)[0][0]
                    for multi_match_id in multi_match_ids
                ]
                given_neuron_ids.append(dat.neurons.unit_ids[neuron_idx])
        match_set -= {-1}  # remove the unmatched neurons
        match_set = np.array(list(match_set))

        # get unit_ids of matched neurons
        if multi_match_ids is not None:
            neuron_ids = given_neuron_ids
        elif multi_match_n is not None:
            random_state = np.random.get_state()
            if multi_match_base_seed is not None:
                np.random.seed(
                    multi_match_base_seed * multi_match_n
                )  # avoid nesting by making seed dependent on number of neurons
            assert (
                len(match_set) >= exclude_multi_match_n + multi_match_n
            ), "After excluding {} neurons, there are not {} matched neurons left".format(
                exclude_multi_match_n, multi_match_n
            )
            match_subset = np.random.choice(
                match_set, size=exclude_multi_match_n + multi_match_n, replace=False
            )[exclude_multi_match_n:]
            neuron_ids = [
                pdsi[np.isin(munit_ids, match_subset)]
                for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)
            ]
            np.random.set_state(random_state)
        else:
            neuron_ids = [
                pdsi[np.isin(munit_ids, match_set)]
                for munit_ids, pdsi in zip(multi_unit_ids, per_data_set_ids)
            ]

        # generate single dataloaders
        dls = OrderedDict({})
        keys = [tier] if tier else ["train", "validation", "test"]
        for key in keys:
            dls[key] = OrderedDict({})

        image_ids = [image_ids] if image_ids is None else image_ids
        for path, neuron_id, image_id in zip_longest(
            paths, neuron_ids, image_ids, fillvalue=None
        ):
            data_key, loaders = Lurz2021Dataset.static_loader(
                path,
                batch_size,
                areas=areas,
                layers=layers,
                cuda=cuda,
                tier=tier,
                get_key=True,
                neuron_ids=neuron_id,
                neuron_n=None,
                neuron_base_seed=None,
                image_ids=image_id,
                image_n=image_n,
                image_base_seed=image_base_seed,
                normalize=normalize,
                include_behavior=include_behavior,
                exclude=exclude,
                select_input_channel=select_input_channel,
                file_tree=True,
                return_test_sampler=return_test_sampler,
                oracle_condition=oracle_condition,
            )
            for k in dls:
                dls[k][data_key] = loaders[k]

        return dls


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
        batch_size: int = 64,
        latent_space_dimension: int = 128,
        device: str = None,
    ) -> None:
        """Empty Latent Vector available as self.vector.
        Args:
            batch_size (int, optional): Batch size. Defaults to 64.
            latent_space_dimension (int, optional): Latent space dimension. Defaults to 128.
            device (str, optional): torch device, if None tries cuda. Defaults to None.
        """
        self.batch_size = batch_size
        self.latent_space_dimension = latent_space_dimension
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.vector = torch.empty(
            size=(self.batch_size, self.latent_space_dimension),
            dtype=torch.float,
            device=self.device,
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
    batch_size: int = 64,
    latent_space_dimension: int = 128,
    device: str = None,
    low: float = -2,
    high: float = 2,
) -> torch.Tensor:
    """Returns a latent tensor with gradient drawn form uniform distribution.

    Args:
        batch_size (int, optional): batchsize. Defaults to 64.
        latent_space_dimension (int, optional): Dimensionality of Vectors. Defaults to 128.
        device (str, optional): Device, if none tries to default to cuda. Defaults to None.
        low (float, optional): low of uniform distribution to sample from. Defaults to 0.
        high (float, optional): high of uniform distribution to sample from. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of size (num_vectors, latent_space_dimension) sampled
            from normal distribution with mean: mean and std: std.
    """
    tensor = LatentVector(batch_size, latent_space_dimension, device)
    return torch.nn.init.uniform_(tensor.vector, low, high)


if __name__ == "__main__":
    pass
