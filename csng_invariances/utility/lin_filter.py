"""Provide different linear filters to estimate a linear receptive field."""

import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from utility.data_helpers import make_directories
from datetime import datetime
from rich import print
from rich.progress import track
import csv


figure_sizes = {
    "full": (8, 5.6),
    "half": (5.4, 3.8),
}


def __mkdir():
    """Make expected directories.

    Returns:
        tuple: Tuple of pathlib.Paths for figure and report."""
    dirs = make_directories()
    date_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    figure_dir = dirs[8] / date_time
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir = dirs[7] / date_time
    report_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, report_dir


def __hyperparametersearchplot(
    corrs, report, display, figure_dir, report_dir, neuron_counter=None
):
    """Create hyperparametersearch plots.

    Args:
        corrs (dict): Dictionary of regularization factor and correlation.
        report (bool): If true reports are stored.
        display (bool): If plots are shown.
        figure_dir (pathlib.Path): Path to figures.
        report_dir (pathlib.Path): Path to reports.
        neuron_counter (int, optional): Current examined neuron. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=figure_sizes["half"])
    ax.scatter(corrs.keys(), corrs.values())
    ax.set_title("Filter hyperparametersearch")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization factor")
    ax.set_ylabel("Correlation [%]")
    if report is True:
        if neuron_counter is None:
            figure_filename = "filter_hyperparametersearch.svg"
            report_filename = "filter_hyperparametersearch.csv"
        else:
            figure_filename = f"filter_hyperparametersearch_neuron_{neuron_counter}.svg"
            report_filename = f"filter_hyperparametersearch-neuron_{neuron_counter}.csv"
        plt.savefig(figure_dir / figure_filename, bbox_inches="tight")
        plt.close()
        with open(report_dir / report_filename, "w") as file:
            for key in corrs:
                file.write("%s,%s\n" % (key, corrs[key]))
        print(f"Stored images in {figure_dir} and report in {report_dir}")
    if display is True:
        plt.show()


def conduct_global_hyperparametersearch(
    TrainFilter, ValFilter, reg_factors, report=True, display=False
):
    """Conduct hyperparametersearch for filter computation

    Args:
        Trainfilter (filter object): filter object based on train dataset
        Valfilter (filter object): filter object based on validation dataset
        reg_factors (list): List of regularization factors
        report (bool, optional): Report option. Defaults to True.
        display (bool, optional): Graph display option. Defaults to False.

    Returns:
        tuple: Tuple of best parameter and dictionary of reg_factors and correlations
    """
    # Hyperparametersearch
    corrs = {}
    print("Beginning hyperparametersearch.")
    for reg_factor in track(reg_factors):
        filter = TrainFilter.train(reg_factor)
        _, corr = ValFilter.predict(filter)
        corrs[reg_factor] = corr
    parameter = max(corrs, key=corrs.get)
    print("Hyperparametersearch concluded.")

    # Make dirs
    figure_dir, report_dir = __mkdir()

    # Create figure
    __hyperparametersearchplot(corrs, report, display, figure_dir, report_dir)

    return parameter, corrs


def conduct_individual_hyperparametersearch(
    TrainFilter, ValFilter, reg_factors, report=True
):
    """Conduct hyperparametersearch with individual regularization factors.

    A hyperparametersearch is conducted from the list of regularization factors.
    For each neuron the regularization factor from reg_factors yielding the highest
    pearson correlation coeffienct is picked.

    Args:
        TrainFilter (Filter object): Linear filters of training data.
        ValFilter (Filter object): Linear filters of validation data.
        reg_factors (list): List of regularization factors to test.
        report (bool, optional): If true, regularization factors and single neuron
            correlations are saved. Defaults to True.

    Returns:
        np.array: 2D array of neuron, regularization factor and single neuron
            correlation.
    """
    print("Beginning Hyperparametersearch.")
    hyperparameters = []
    for neuron in track(range(TrainFilter.neuron_count)):
        corrs = {}
        for reg_factor in reg_factors:
            reg_factor = [reg_factor for i in range(TrainFilter.neuron_count)]
            filter = TrainFilter.train(reg_factor)
            predictions, _ = ValFilter.predict(filter)
            prediction = predictions[:, neuron]
            response = ValFilter.responses[:, neuron]
            correlation = np.corrcoef(response, prediction)[0, 1]
            corrs[reg_factor[0]] = correlation
        best_reg_factor = max(corrs, key=corrs.get)
        hyperparameters.append([neuron, best_reg_factor, corrs[best_reg_factor]])
    print("Hyperparametersearch concluded.")

    if report is True:
        _, report_dir = __mkdir()
        with open(report_dir / "Induvidual_hyperparameters.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(hyperparameters)

    return np.asarray(hyperparameters)


def _reshape_filter_2d(fil):
    """Reshape filter to 2D representation

    Args:
        fil (np.array): 4D representation of filter

    Returns:
        np.array: 2D representation of array
    """
    assert len(fil.shape) == 4, f"Filter was expected to be 4D but is {fil.shape}"
    neuron_count = fil.shape[0]
    dim1 = fil.shape[2]
    dim2 = fil.shape[3]
    fil = fil.squeeze()
    fil = fil.reshape(neuron_count, dim1 * dim2)
    fil = np.moveaxis(fil, [0, 1], [1, 0])
    return fil


def _make_dirs():
    """Make required directory structure.

    Returns:
        tuple: Tuple of pathlib.Paths for figures and reports
    """
    dirs = make_directories()
    date_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    figure_dir = dirs[8] / date_time
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir = dirs[7] / date_time
    report_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, report_dir


class Filter:
    """Class of linear filters for lin. receptive field approximation."""

    def __init__(self, images, responses, reg_type, reg_factor):
        """Infilterntiates Class

        Args:
            images (np.array): image data
            responses (np.array): response data
            reg_type (str, optional): Type of regularization used to compute filter.
                Options are:
                    - "laplace regularied",
                    - "ridge regularized",
                    - "whitened",
                    - "raw".
                Defaults to "laplace regularized".
            reg_factor (optional): Regularization factor.
        """
        # Instanatiate attributes from arguments
        self.images = images
        self.image_count = images.shape[0]
        self.channels = images.shape[1]
        self.dim1 = images.shape[2]
        self.dim2 = images.shape[3]
        self.responses = responses
        self.neuron_count = self.responses.shape[1]
        self.reg_factor = reg_factor
        self.reg_type = reg_type
        assert self.reg_type in set(
            ["laplace regularized", "ridge regularized", "whitened", "raw"]
        ), "No valid type option. Options are 'laplace regularized', \
            'ridge regularized', 'whitened' and 'raw'."
        # Laplace regularized case
        if self.reg_type == "laplace regularized":
            self._compute_filter = self._laplace_regularized_filter
        # ridge regularized case
        elif self.reg_type == "ridge regularized":
            self._compute_filter = self._ridge_regularized_filter
        # whitened case
        elif self.reg_type == "whitened":
            self._compute_filter = self._whitened_filter
        # base case
        else:
            self._compute_filter = self._normal_filter

        # instantiate attributes
        self.fil = None
        self.prediction = None
        self.corr = None

    def train(self):
        None

    def predict(self, fil=None):
        """Predict response on filter.

        If no filter is passed, the computed filter is used.

        Args:
            fil (np.array): Filter to use for prediction. Defaults to None.

        Returns:
            tuple: tuple of predictions and correlation
        """
        fil = self._handle_predict_parsing(fil)
        self._image_2d()
        self.prediction = np.asarray(np.matmul(self.images, fil))
        self.corr = np.corrcoef(self.prediction.flatten(), self.responses.flatten())[
            0, 1
        ]
        return self.prediction, self.corr

    def evaluate(self, fil=None, output=False):
        """Generate fit report of Filter.

        If no filter is passed, the computed filter is used.

        Args:
            fil (np.array): Filter to use for report. Defaults to None.
            output (bool): If true images of lin. receptive fields and their correlation
                are depicted. Defaults to False.

        Returns:
            dict: Dictionary of Neurons and Correlations
        """
        computed_prediction = self._handle_evaluate_parsing(fil)
        fil = _reshape_filter_2d(fil)
        # Make directories
        figure_dir, report_dir = _make_dirs()
        # Make report
        self.neural_correlations = {}
        print("Begin reporting procedure.")
        if output is True:
            print(f"Stored images at {figure_dir} and report at {report_dir}.")
        for neuron in track(range(computed_prediction.shape[1])):
            corr = np.corrcoef(
                computed_prediction[:, neuron], self.responses[:, neuron]
            )[0, 1]
            if output is True:
                fig, ax = plt.subplots(figsize=figure_sizes["half"])
                im = ax.imshow(fil[:, neuron].reshape(self.dim1, self.dim2))
                ax.set_title(f"Neuron: {neuron} | Correlation: {round(corr*100,2)}%")
                fig.colorbar(im)
                plt.savefig(
                    figure_dir / f"Spike-triggered_Average_{neuron}.svg",
                    bbox_inches="tight",
                )
                plt.close()
            self.neural_correlations[neuron] = corr
        if output is True:
            with open(
                report_dir / "Spike-triggered_Average_Correlations.csv", "w"
            ) as file:
                for key in self.neural_correlations:
                    file.write("%s,%s\n" % (key, self.neural_correlations[key]))
        print("Reporting procedure concluded.")
        return self.neural_correlations

    def select_neurons(self):
        # TODO neuron selection process
        pass

    def _shape_printer(self):
        """Print shape related information for debugging."""
        print(
            f"The current image shape is {self.images.shape}\
              \nand the response shape is {self.responses.shape}."
        )
        if self.fil is None:
            print("No filter has yet been computed.")
        else:
            print(f"The current filter shape is {self.fil.shape}.")
        if self.prediction is None:
            print("No predictions have been computed yet.")
        else:
            print(f"The current prediction shape is {self.prediction.shape}")

    def _image_2d(self):
        """Reshape image dataset to 2D representation."""
        self.images = self.images.reshape(self.image_count, self.dim1 * self.dim2)

    def _fil_2d(self):
        """Reshape filter dataset to 2D representation."""
        if len(self.fil.shape) == 4:
            assert (
                len(self.fil.shape) == 4
            ), f"The filter was expected to be 4D but is of shape {self.fil.shape}."
            self.fil = self.fil.squeeze()
            self.fil = self.fil.reshape(self.neuron_count, self.dim1 * self.dim2)
            self.fil = np.moveaxis(self.fil, [0, 1], [1, 0])

    def _fil_4d(self):
        """Reshape filter dataset to 4D representation."""
        if len(self.fil.shape) == 2:
            self.fil = np.moveaxis(self.fil, [0, 1], [1, 0])
            self.fil = self.fil.reshape(
                self.neuron_count,
                self.channels,
                self.dim1,
                self.dim2,
            )

    def _normal_filter(self, responses, **kwargs):
        """Compute Spike-triggered Average.

        Args:
            responses (np.array): 2D representation of the response data.

        Returns:
            np.array: 2D representation of linear filters.  Filters are flattened.
        """
        self._image_2d()
        fil = np.matmul(self.images.T, responses)
        return fil

    def _whitened_filter(self, responses, **kwargs):
        """Compute whitened Spike-triggered Average.

        Args:
            responses (np.array): 2D representation of the response data.

        Returns:
            np.array: 2D representation of linear filters. Filters are flattened.
        """
        self._image_2d()
        fil = np.matmul(
            pinv(np.matmul(self.images.T, self.images)),
            np.matmul(self.images.T, responses),
        )
        return fil

    def _ridge_regularized_filter(self, responses, reg_factor):
        """Compute ridge regularized spike-triggered average.

        Args:
            responses (np.array): 2D representation of the response data.
            reg_factor (float): regularization factor.

        Returns:
            np.array: 2D representation of linear filters. Filters are flattened.
        """
        self._image_2d()
        fil = np.matmul(
            pinv(
                np.matmul(self.images.T, self.images)
                + np.dot(reg_factor, np.identity(self.dim1 * self.dim2))
            ),
            np.matmul(self.images.T, responses),
        )
        return fil

    def _laplace_regularized_filter(self, responses, reg_factor):
        """Compute laplace regularized spike-triggered average

        Args:
            responses (np.array): 2D representation of the response data.
            reg_factor (float): Regularization factor.

        Returns:
            np.array: 2D representation of linear filter. Filters are flattened.
        """

        def __laplaceBias(sizex, sizey):
            """Generate matrix based on discrete laplace operator with sizex * sizey.

            Args:
                sizex (int): x-dimension of to be regularized object
                sizey (int): y-dimension of to be regularized object

            Returns:
                np.matrix: matrix based on laplace operator
            """
            S = np.zeros((sizex * sizey, sizex * sizey))
            for x in range(0, sizex):
                for y in range(0, sizey):
                    norm = np.mat(np.zeros((sizex, sizey)))
                    norm[x, y] = 4
                    if x > 0:
                        norm[x - 1, y] = -1
                    if x < sizex - 1:
                        norm[x + 1, y] = -1
                    if y > 0:
                        norm[x, y - 1] = -1
                    if y < sizey - 1:
                        norm[x, y + 1] = -1
                    S[x * sizex + y, :] = norm.flatten()
            S = np.mat(S)
            return S

        self._image_2d()
        laplace = __laplaceBias(self.dim1, self.dim2)
        ti = np.vstack((self.images, np.dot(reg_factor, laplace)))
        ts = np.vstack(
            (
                responses,
                np.zeros((self.images.shape[1], responses.shape[1])),
            )
        )
        fil = np.asarray(pinv(ti.T * ti) * ti.T * ts)
        return fil

    def _handle_train_parsing(self, reg_factor):
        """Handle parsing of a regularization factor during `train()`.

        Args:
            reg_factor (float): Regularization factor.
        """
        if reg_factor is None:
            reg_factor = self.reg_factor
        return reg_factor

    def _handle_predict_parsing(self, fil):
        """Handle parsing of filters during `predict()`

        Args:
            fil (np.array): 4D representation of linear filters.
        """
        if fil is None:
            if self.fil is None:
                self.train()
            self._fil_2d()
            fil = self.fil
        if len(fil.shape) == 4:
            fil = _reshape_filter_2d(fil)
        return fil

    def _handle_evaluate_parsing(self, fil):
        """Handle parsing of filters during `evaluate()`.

        Args:
            fil (np.array): 4D-representation of filters.

        Returns:
            np.array: predictions.
        """
        if fil is None:
            if self.fil is None:
                self.fil = self.train()
            if self.prediction is None:
                self.predict()
            fil = self.fil
            computed_prediction = self.prediction
        else:
            if len(fil.shape) == 4:
                fil = _reshape_filter_2d(fil)
            computed_prediction, _ = self.predict(fil)

        return computed_prediction


class GlobalRegularizationFilter(Filter):
    """Global regularized linear filter class.

    Class of linear filters with global regularization factor applied."""

    def __init__(
        self, images, responses, reg_type="laplace regularized", reg_factor=10
    ):
        super().__init__(images, responses, reg_type, reg_factor)

    def train(self, reg_factor=None):
        """Compute linear filters fitting images to neural responses.

        Args:
            reg_factor (float, optional): Regularization factor. Defaults to None.

        Returns:
            np.array: 4D representation of filter.
        """
        reg_factor = self._handle_train_parsing(reg_factor)
        self._image_2d()
        self.fil = self._compute_filter(responses=self.responses, reg_factor=reg_factor)
        self._fil_4d()
        return self.fil

    def predict(self, fil=None):
        return super().predict(fil=fil)

    def evaluate(self, fil=None, output=False):
        return super().evaluate(fil=fil, output=output)

    def select_neurons(self):
        return super().select_neurons()


class IndividualRegularizationFilter(Filter):
    """Individually regularized linear filter class.

    Class of linear filters with individual regularization factors applied."""

    def __init__(
        self, images, responses, reg_type="laplace regularized", reg_factor=None
    ):
        super().__init__(images, responses, reg_type, reg_factor)
        if reg_factor is None:
            self.reg_factor = [10 for i in range(self.responses.shape[1])]

    def train(self, reg_factors=None):
        """Compute linear filters with individual regularization.

        Args:
            reg_factors (list, optional): List of regularization factors
                (one per neuron). Defaults to None.

        Returns:
            np.array: 2D representation of linear filters. Filters are flattened.
        """
        reg_factors = self._handle_train_parsing(reg_factors)
        filters = np.empty((self.dim1 * self.dim2, self.neuron_count))
        for neuron, reg_factor in zip(range(self.neuron_count), reg_factors):
            self._image_2d()
            response = self.responses[:, neuron].reshape(self.image_count, 1)
            fil = self._compute_filter(responses=response, reg_factor=reg_factor)
            filters[:, neuron] = fil.squeeze()
        self.fil = filters
        self._fil_4d()
        return self.fil

    def predict(self, fil=None):
        return super().predict(fil=fil)

    def evaluate(self, fil=None, output=False):
        return super().evaluate(fil=fil, output=output)

    def select_neurons(self):
        return super().select_neurons()


if __name__ == "__main__":
    pass
