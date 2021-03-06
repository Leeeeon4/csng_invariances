"""Provide different discriminator models."""

from os import device_encoding
from typing import Tuple
import torch
import numpy as np
import copy
import requests

from torch import nn
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from pathlib import Path

# private fork of neuralpredictors
from csng_invariances._neuralpredictors.layers.readouts import (
    MultipleFullGaussian2d,
    MultiplePointPooled2d,
    MultipleFullSXF,
)
from csng_invariances._neuralpredictors.layers.cores import (
    TransferLearningCore,
    SE2dCore,
)

from csng_invariances.data._data_helpers import (
    unpack_data_info,
    load_configs,
    adapt_config_to_machine,
)

from csng_invariances.data.datasets import Lurz2021Dataset

from csng_invariances.layers.custom_layers import (
    DifferenceOfGaussiangLayer,
    HSMCorticalBlock,
)


def download_pretrained_lurz_model():
    """Download the pretrained lurz model.

    Downloads the model into the models directory.
    Only the core of the model is pretrained, the readout has to be trained for
    the specific dataset.

    Returns
        pathlib.Path: Path to model
    """
    # TODO deprecate, use data.datasets instead.
    print("Downloading pretrained Lurz model.")
    url = "https://github.com/sinzlab/Lurz_2020_code/raw/main/notebooks/models/transfer_model.pth.tar"
    answer = requests.get(url, allow_redirects=True)
    # make directory
    file_dir = Path.cwd() / "models" / "external" / "lurz2020"
    file_dir.mkdir(parents=True, exist_ok=True)
    # save file
    with open(file_dir / "transfer_model.pth.tar", "wb") as fd:
        fd.write(answer.content)
    print("Done")
    return file_dir / "transfer_model.pth.tar"


def get_core_trained_model(dataloaders):
    """Load pretrained Lurz model.

    Args:
        dataloaders (OrderedDict): dictionary of dictionaries where the first
            level keys are 'train', 'validation', and 'test', and second level
            keys are data_keys.

    Returns:
        torch.nn.Module: pretrained model
    """
    # TODO deprecate as model_config is not passed.
    model_config = {
        "init_mu_range": 0.55,
        "init_sigma": 0.4,
        "input_kern": 15,
        "hidden_kern": 13,
        "gamma_input": 1.0,
        "grid_mean_predictor": {
            "type": "cortex",
            "input_dimensions": 2,
            "hidden_layers": 0,
            "hidden_features": 0,
            "final_tanh": False,
        },
        "gamma_readout": 2.439,
    }

    model = se2d_fullgaussian2d(**model_config, dataloaders=dataloaders, seed=1)
    # Download pretrained model if not there

    model_path = Path.cwd() / "models" / "external" / "lurz2020"
    if (model_path / "transfer_model.pth.tar").is_file() is False:
        download_pretrained_lurz_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    transfer_model = torch.load(
        model_path / "transfer_model.pth.tar",
        map_location=device,
    )
    model.load_state_dict(transfer_model, strict=False)

    return model


class Encoder(nn.Module):
    """This class was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX."""

    def __init__(self, core, readout, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset

    def forward(self, x, data_key=None, detach_core=False, **kwargs) -> torch.Tensor:
        x = self.core(x)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"])
        else:
            x = self.readout(x, data_key=data_key)
        return nn.functional.elu(x + self.offset) + 1


def se2d_fullgaussian2d(
    dataloaders,
    seed,
    elu_offset=0,
    data_info=None,
    transfer_state_dict=None,
    # core args
    hidden_channels=64,
    input_kern=9,
    hidden_kern=7,
    layers=4,
    gamma_input=6.3831,
    skip=0,
    bias=False,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    stack=-1,
    se_reduction=32,
    n_se_blocks=0,
    depth_separable=True,
    linear=False,
    # readout args
    init_mu_range=0.3,
    init_sigma=0.1,
    readout_bias=True,
    gamma_readout=0.0076,
    gauss_type="full",
    grid_mean_predictor={
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 30,
        "final_tanh": True,
    },
    share_features=False,
    share_grid=False,
    share_transform=False,
    init_noise=1e-3,
    init_transform_scale=0.2,
):
    """Model class of a SE2dCore and a Gaussian readout.

    This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX.

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_grid: whether to share the grid between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_transform: whether to share the transform from the grid_mean_predictor between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        init_noise: noise for initialization of weights
        init_transform_scale: scale of the weights of the randomly intialized grid_mean_predictor network
        all other args: See Documentation of SE2dCore in csng_invariances.neuralpredictors.layers.cores and
            FullGaussian2d in csng_invariances.neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """
    if transfer_state_dict is not None:
        print(
            "Transfer state_dict given. This will only have an effect in the bayesian hypersearch. See: TrainedModelBayesianTransfer "
        )

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")
        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {}
            for k, v in dataloaders.items():
                # real data
                try:
                    if v.dataset.neurons.animal_ids[0] != 0:
                        source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                            :, :input_dim
                        ]
                    # simulated data -> get random linear non-degenerate transform of true positions
                    else:
                        source_grid_true = v.dataset.neurons.center[:, :input_dim]
                        det = 0.0
                        loops = 0
                        grid_bias = np.random.rand(2) * 3
                        while det < 5.0 and loops < 100:
                            matrix = np.random.rand(2, 2) * 3
                            det = np.linalg.det(matrix)
                            loops += 1
                        assert det > 5.0, "Did not find a non-degenerate matrix"
                        source_grids[k] = np.add(
                            (matrix @ source_grid_true.T).T, grid_bias
                        )
                except FileNotFoundError:
                    print(
                        "Dataset type is not recognized to be from Baylor College of Medicine."
                    )
                    source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                        :, :input_dim
                    ]
        elif grid_mean_predictor_type == "shared":
            pass
        else:
            raise ValueError(
                "Grid mean predictor type {} not understood.".format(
                    grid_mean_predictor_type
                )
            )

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {
            k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()
        }
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        share_features=share_features,
        share_grid=share_grid,
        share_transform=share_transform,
        shared_match_ids=shared_match_ids,
        init_noise=init_noise,
        init_transform_scale=init_transform_scale,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se2d_pointpooled(
    dataloaders,
    seed,
    elu_offset=0,
    data_info=None,
    # core args
    hidden_channels=64,
    input_kern=9,  # core args
    hidden_kern=7,
    layers=4,
    gamma_input=46.402,
    bias=False,
    skip=0,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    stack=-1,
    se_reduction=32,
    n_se_blocks=0,
    depth_separable=True,
    linear=False,
    # readout args
    pool_steps=2,
    pool_kern=3,
    readout_bias=True,
    gamma_readout=0.0207,
    init_range=0.2,
):
    """Model class of a SE2dCore and a pointpooled (spatial transformer) readout.

    This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX.

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of SE2dCore in csng_invariances.neuralpredictors.layers.cores and
            PointPooled2D in csng_invariances.neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        bias=bias,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultiplePointPooled2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        pool_steps=pool_steps,
        pool_kern=pool_kern,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        init_range=init_range,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se2d_fullSXF(
    dataloaders,
    seed,
    elu_offset=0,
    data_info=None,
    transfer_state_dict=None,
    # core args
    hidden_channels=64,
    input_kern=9,
    hidden_kern=7,
    layers=4,
    gamma_input=6.3831,
    skip=0,
    bias=False,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    stack=-1,
    se_reduction=32,
    n_se_blocks=0,
    depth_separable=True,
    linear=False,
    init_noise=4.1232e-05,
    normalize=False,
    readout_bias=True,
    gamma_readout=0.0076,
    share_features=False,
):
    """Model class of a SE2dCore and a factorized (sxf) readout

    This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX.

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of SE2dCore in csng_invariances.neuralpredictors.layers.cores and
            fullSXF in csng_invariances.neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if transfer_state_dict is not None:
        print(
            "Transfer state_dict given. This will only have an effect in the bayesian hypersearch. See: TrainedModelBayesianTransfer "
        )
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    shared_match_ids = None
    if share_features:
        shared_match_ids = {
            k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()
        }
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultipleFullSXF(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_noise=init_noise,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        normalize=normalize,
        share_features=share_features,
        shared_match_ids=shared_match_ids,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def taskdriven_fullgaussian2d(
    dataloaders,
    seed,
    elu_offset=0,
    data_info=None,
    # core args
    tl_model_name="vgg16",
    layers=4,
    pretrained=True,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    # readout args
    init_mu_range=0.3,
    init_sigma=0.1,
    readout_bias=True,
    gamma_readout=0.0076,
    gauss_type="full",
    grid_mean_predictor={
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 30,
        "final_tanh": True,
    },
    share_features=False,
    share_grid=False,
    share_transform=False,
    init_noise=1e-3,
    init_transform_scale=0.2,
):
    """Model class of a task-driven transfer core and a Gaussian readout.

    This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    DATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX.

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_grid: whether to share the grid between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_transform: whether to share the transform from the grid_mean_predictor between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        init_noise: noise for initialization of weights
        init_transform_scale: scale of the weights of the randomly intialized grid_mean_predictor network
        all other args: See Documentation of TransferLearningCore in csng_invariances.neuralpredictors.layers.cores and
            FullGaussian2d in csng_invariances.neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")
        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {}
            for k, v in dataloaders.items():
                # real data
                try:
                    if v.dataset.neurons.animal_ids[0] != 0:
                        source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                            :, :input_dim
                        ]
                    # simulated data -> get random linear non-degenerate transform of true positions
                    else:
                        source_grid_true = v.dataset.neurons.center[:, :input_dim]
                        det = 0.0
                        loops = 0
                        grid_bias = np.random.rand(2) * 3
                        while det < 5.0 and loops < 100:
                            matrix = np.random.rand(2, 2) * 3
                            det = np.linalg.det(matrix)
                            loops += 1
                        assert det > 5.0, "Did not find a non-degenerate matrix"
                        source_grids[k] = np.add(
                            (matrix @ source_grid_true.T).T, grid_bias
                        )
                except FileNotFoundError:
                    print(
                        "Dataset type is not recognized to be from Baylor College of Medicine."
                    )
                    source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                        :, :input_dim
                    ]
        elif grid_mean_predictor_type == "shared":
            pass
        else:
            raise ValueError(
                "Grid mean predictor type {} not understood.".format(
                    grid_mean_predictor_type
                )
            )

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {
            k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()
        }
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    set_random_seed(seed)

    core = TransferLearningCore(
        input_channels=core_input_channels,
        tl_model_name=tl_model_name,
        layers=layers,
        pretrained=pretrained,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    readout = MultipleFullGaussian2d(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        share_features=share_features,
        share_grid=share_grid,
        shared_match_ids=shared_match_ids,
        share_transform=share_transform,
        init_noise=init_noise,
        init_transform_scale=init_transform_scale,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def taskdriven_fullSXF(
    dataloaders,
    seed,
    elu_offset=0,
    data_info=None,
    # core args
    tl_model_name="vgg16",
    layers=4,
    pretrained=True,
    final_batchnorm=True,
    final_nonlinearity=True,
    momentum=0.1,
    fine_tune=False,
    # readout args
    init_noise=4.1232e-05,
    normalize=False,
    readout_bias=True,
    gamma_readout=0.0076,
    share_features=False,
):
    """
    Model class of a task-driven transfer core and a factorized (sxf) readout

    This function was provided by Lurz et al. ICLR 2021: GENERALIZATION IN
    ATA-DRIVEN MODELS OF PRIMARY VISUAL CORTEX.

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of TransferLearningCore  in csng_invariances.neuralpredictors.layers.cores and
            fullSXF in csng_invariances.neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    shared_match_ids = None
    if share_features:
        shared_match_ids = {
            k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()
        }
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    set_random_seed(seed)

    core = TransferLearningCore(
        input_channels=core_input_channels,
        tl_model_name=tl_model_name,
        layers=layers,
        pretrained=pretrained,
        final_batchnorm=final_batchnorm,
        final_nonlinearity=final_nonlinearity,
        momentum=momentum,
        fine_tune=fine_tune,
    )

    readout = MultipleFullSXF(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_noise=init_noise,
        bias=readout_bias,
        gamma_readout=gamma_readout,
        normalize=normalize,
        share_features=share_features,
        shared_match_ids=shared_match_ids,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def load_encoding_model(model_directory):
    """Load pretrained encoding model.

    Loads an encoding model which was pretrained for a specific dataset. The
    model is return in eval model. I.e. dropout and other things are deactivated
    which cause inference.

    Args:
        model_directory (str): path to model directory to load from.

    Returns:
        model: Trained encoding model in evaluation state.
    """
    configs = load_configs(model_directory)
    print(f"Model was trained on {configs['trainer_config']['device']}.\n")
    configs = adapt_config_to_machine(configs)
    dataloaders = Lurz2021Dataset.static_loaders(**configs["dataset_config"])
    model = se2d_fullgaussian2d(
        **configs["model_config"],
        dataloaders=dataloaders,
        seed=configs["dataset_config"]["seed"],
    )
    model.load_state_dict(
        torch.load(
            Path(model_directory) / "Pretrained_core_readout_lurz.pth",
            map_location=configs["trainer_config"]["device"],
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model runs in evaluation mode on {device}.")
    model.eval()
    return model


class HSMModel(nn.Module):
    """HSM Model as presented by Antolik et al. 2016: Model Constrained by Visual
    Hierarchy Improves Prediction of Neural Responses to Natural Scenes, available
    at: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927
    """

    def __init__(
        self,
        images_size: Tuple[int, int, int, int],
        num_neurons: int,
        num_lgn_units: int = 9,
        hidden_units_fraction: float = 0.2,
        device: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.batch_size, self.channels, self.height, self.width = images_size
        self.num_neurons = num_neurons
        self.num_lgn_units = num_lgn_units
        self.hidden_units_fraction = hidden_units_fraction
        # self.difference_of_gaussians_layer = DifferenceOfGaussiangLayer(
        #     self.num_lgn_units, self.width, self.height, *args, **kwargs
        # )
        # self.cortical_layer_one = HSMCorticalBlock(
        #     self.num_lgn_units,
        #     int(self.num_neurons * self.hidden_units_fraction),
        #     *args,
        #     **kwargs,
        # )
        # self.cortical_layer_two = HSMCorticalBlock(
        #     int(self.num_neurons * self.hidden_units_fraction),
        #     self.num_neurons,
        #     *args,
        #     **kwargs,
        # )
        # self.stack = nn.Sequential(
        #     self.difference_of_gaussians_layer,
        #     self.cortical_layer_one,
        #     self.cortical_layer_two,
        # )
        self.stack = nn.Sequential(
            DifferenceOfGaussiangLayer(
                self.num_lgn_units, self.width, self.height, *args, **kwargs
            ),
            HSMCorticalBlock(
                self.num_lgn_units,
                int(self.num_neurons * self.hidden_units_fraction),
                *args,
                **kwargs,
            ),
            HSMCorticalBlock(
                int(self.num_neurons * self.hidden_units_fraction),
                self.num_neurons,
                *args,
                **kwargs,
            ),
        )
        self.stack.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        return x


if __name__ == "__main__":
    pass
