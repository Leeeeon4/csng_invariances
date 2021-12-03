"""Module providing functions related to the generation of most exciting images."""


def mei_generation():
    # TODO finalize mei computation
    from csng_invariances.encoding import load_encoding_model
    from csng_invariances.metrics_statistics.correlations import (
        compute_single_neuron_correlations_encoding_model as dnn_corrs,
    )
    from csng_invariances.metrics_statistics.correlations import (
        load_single_neuron_correlations_linear_filter,
    )
    from csng_invariances.training.mei import mei
    from csng_invariances.layers.loss_function import SelectedNeuronActivation
    from csng_invariances.data.datasets import (
        gaussian_white_noise_image,
        Lurz2021Dataset,
    )
    from csng_invariances.metrics_statistics.select_neurons import select_neurons, score
    from csng_invariances.data._data_helpers import load_configs

    model_directory = "/home/leon/csng_invariances/models/encoding/2021-11-30_15:15:03"
    encoding_model = load_encoding_model(model_directory)
    configs = load_configs(model_directory)
    ds = Lurz2021Dataset(dataset_config=configs["dataset_config"])
    images, responses = ds.get_dataset()
    criterion = SelectedNeuronActivation()
    gwni = gaussian_white_noise_image(
        size=(1, images.shape[1], images.shape[2], images.shape[3])
    )
    dnn_single_neuron_correlations = dnn_corrs(
        encoding_model,
        images,
        responses,
        batch_size=configs["dataset_config"]["batch_size"],
    )
    linear_filter_single_neuron_correlations = load_single_neuron_correlations_linear_filter(
        "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-30_15:15:09/Correlations.csv"
    )
    selection_score = score(
        dnn_single_neuron_correlations, linear_filter_single_neuron_correlations
    )
    select_neuron_indicies = select_neurons(selection_score, 5)
    meis = mei(criterion, encoding_model, gwni, select_neuron_indicies, epochs=2000)


if __name__ == "__main__":
    mei_generation()
