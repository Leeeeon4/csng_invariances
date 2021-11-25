"""Module providing functions related to the generation of exciting images."""


def mei_generation():
    from csng_invariances.encoding import (
        load_encoding_model
    )
    from csng_invariances.encoding import get_single_neuron_correlation as dnn_corrs
    from csng_invariances.training.mei import mei
    from csng_invariances.losses.loss_modules import SelectedNeuronActivation
    from csng_invariances.data.datasets import GaussianWhiteNoiseImage, Lurz2021Dataset
    from csng_invariances.select_neurons import select_neurons, score
    from csng_invariances.data._data_helpers import load_configs

    model_directory = (
        "/Users/leongorissen/csng_invariances/models/encoding/2021-11-12_17:00:58"
    )
    encoding_model = load_encoding_model(model_directory)
    configs = load_configs(model_directory)
    ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
    images, responses = ds.get_dataset()
    criterion = SelectedNeuronActivation()
    gaussian_white_noise_image = GaussianWhiteNoiseImage(
        size=(1, images.shape[1], images.shape[2], images.shape[3])
    )
    dnn_single_neuron_correlations = dnn_corrs(
        encoding_model, images, responses, batch_size=configs["dataset_config"]["batch_size"]
    )
    linear_filter_single_neuron_correlations = 
    selection_score = score(
        dnn_single_neuron_correlations, linear_filter_single_neuron_correlations
    )
    select_neuron_indicies = select_neurons(selection_score, 5)
    meis = mei()


def generator_generation():
    pass


if __name__ == "__main__":
    pass
