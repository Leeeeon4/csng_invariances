"""Provide different linear filters to estimate a linear receptive field."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from numpy.linalg import pinv
from concurrent.futures import ProcessPoolExecutor
from csng_invariances._utils.utlis import string_time
from rich import print
from rich.progress import track
from pathlib import Path

from csng_invariances.data._data_helpers import scale_tensor_to_0_1 as norm


figure_sizes = {
    "full": (8, 5.6),
    "half": (5.4, 3.8),
}


def _reshape_filter_2d(fil):
    """Reshape filter to 2D representation

    Args:
        fil (np.array): 4D representation of filter

    Returns:
        np.array: 2D representation of array
    """
    # TODO make class method.
    assert len(fil.shape) == 4, f"Filter was expected to be 4D but is {fil.shape}"
    neuron_count = fil.shape[0]
    dim1 = fil.shape[2]
    dim2 = fil.shape[3]
    fil = fil.squeeze()
    fil = fil.reshape(neuron_count, dim1 * dim2)
    fil = np.moveaxis(fil, [0, 1], [1, 0])
    return fil


class Filter:
    """Class of linear filters for lin. receptive field approximation."""

    def __init__(self, images, responses, reg_type, reg_factor):
        """Infilterntiates Class

        Args:
            images (np.array): image self.report_data
            responses (np.array): response self.report_data
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
        self.time = string_time()
        self.model_dir = Path.cwd() / "models" / "linear_filter" / self.time
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = None
        self.figure_dir = None
        self.fil = None
        self.prediction = None
        self.corr = None

    def train(self):
        None

    def predict(self, fil=None, single_neuron_correlation=False):
        """Predict response on filter.

        If no filter is passed, the computed filter is used.

        Args:
            fil (np.array, optional): Filter to use for prediction. Defaults to
                None.
            singel_neuron_correlation (bool, optional): If True, compute single
                neuron correlation. Defaults to False.

        Returns:
            tuple: tuple of predictions and correlation
        """
        # TODO Handle Cuda
        fil = self._handle_predict_parsing(fil)
        self._image_2d()
        self.prediction = np.asarray(np.matmul(self.images.cpu(), fil))
        self.corr = np.corrcoef(
            self.prediction.flatten(), self.responses.cpu().flatten()
        )[0, 1]
        if single_neuron_correlation:
            self.single_neuron_correlations = np.empty(self.neuron_count)
            for neuron in range(self.neuron_count):
                pred = self.prediction[:, neuron]
                resp = self.responses[:, neuron]
                single_corr = np.corrcoef(pred, resp.cpu())[0, 1]
                self.single_neuron_correlations[neuron] = single_corr
        return self.prediction, self.corr

    def evaluate(self, fil=None, reports=True, store_images=False, report_dir=None):
        """Generate fit report of Filter.

        If no filter is passed, the computed filter is used.

        Args:
            fil (np.array, optional): Filter to use for report. Defaults to None.
            reports (bool, optional): If True evaluation reports are stored.
                Defaults to True.
            store_images (bool, optional): If True images of lin. receptive fields
                and their correlation are depicted. Defaults to False.
            report_dir (Path, optional): Path to use to store report. Defaults
                to None.

        Returns:
            dict: Dictionary of Neurons and Correlations
        """
        # TODO handle cuda
        if report_dir is None:
            self.report_dir = Path.cwd() / "reports" / "linear_filter" / self.time
            self.report_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.report_dir = report_dir
        computed_prediction = self._handle_evaluate_parsing(fil)
        fil = _reshape_filter_2d(fil)
        # Make report
        self.neural_correlations = {}
        self.single_neural_correlation_linear_filter = np.empty(self.neuron_count)
        print("Begin reporting procedure.")
        if store_images:
            print(f"Stored images at {self.figure_dir}.")
        for neuron in track(range(computed_prediction.shape[1])):
            corr = np.corrcoef(
                computed_prediction[:, neuron], self.responses[:, neuron].cpu()
            )[0, 1]
            if store_images:
                self.figure_dir = (
                    Path.cwd() / "reports" / "figures" / "linear_filter" / self.time
                )
                self.figure_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=figure_sizes["half"])
                im = ax.imshow(fil[:, neuron].reshape(self.dim1, self.dim2))
                ax.set_title(f"Neuron: {neuron} | Correlation: {round(corr*100,2)}%")
                fig.colorbar(im)
                plt.savefig(
                    self.figure_dir / f"Filter_neuron_{neuron}.svg",
                    bbox_inches="tight",
                )
                plt.close()
            self.neural_correlations[neuron] = corr
            self.single_neural_correlation_linear_filter[neuron] = corr
        self.fil = fil
        self._fil_4d()
        if reports:
            print(f"Reports are stored at {self.report_dir}")
            # TODO save as numpy array to data/preprocessed/linear_filter/correlations.npy.
            with open(self.report_dir / "Correlations.csv", "w") as file:
                for key in self.neural_correlations:
                    file.write("%s,%s\n" % (key, self.neural_correlations[key]))
            print(f"Filter is stored at {self.model_dir}")
            np.save(str(self.model_dir / "evaluated_filter.npy"), self.fil)
            with open(self.model_dir / "readme.txt", "w") as file:
                file.write(
                    (
                        "evaluated_filter.npy contains a 4D representation "
                        "of a linear filter used to estimate the linear "
                        "receptive field of neurons.\nThe dimension are: "
                        f"(neuron, channels, height, width): {self.fil.shape}"
                    )
                )
        print("Reporting procedure concluded.")
        return self.single_neural_correlation_linear_filter

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
        """Reshape image self.report_dataset to 2D representation."""
        self.images = self.images.reshape(self.image_count, self.dim1 * self.dim2)

    def _fil_2d(self):
        """Reshape filter self.report_dataset to 2D representation."""
        if len(self.fil.shape) == 4:
            assert (
                len(self.fil.shape) == 4
            ), f"The filter was expected to be 4D but is of shape {self.fil.shape}."
            self.fil = self.fil.squeeze()
            self.fil = self.fil.reshape(self.neuron_count, self.dim1 * self.dim2)
            self.fil = np.moveaxis(self.fil, [0, 1], [1, 0])

    def _fil_4d(self):
        """Reshape filter self.report_dataset to 4D representation."""
        if len(self.fil.shape) == 2:
            if torch.is_tensor(self.fil):
                self.fil = self.fil.numpy()
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
            responses (np.array): 2D representation of the response self.report_data.

        Returns:
            np.array: 2D representation of linear filters.  Filters are flattened.
        """
        self._image_2d()
        fil = np.matmul(self.images.T, responses)
        return fil

    def _whitened_filter(self, responses, **kwargs):
        """Compute whitened Spike-triggered Average.

        Args:
            responses (np.array): 2D representation of the response self.report_data.

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
            responses (np.array): 2D representation of the response self.report_data.
            reg_factor (float): regularization factor.

        Returns:
            np.array: 2D representation of linear filters. Filters are flattened.
        """
        # TODO Handle Cuda
        self._image_2d()
        fil = np.matmul(
            pinv(
                np.matmul(self.images.cpu().T, self.images.cpu())
                + reg_factor * np.identity(self.dim1 * self.dim2)
            ),
            np.matmul(self.images.cpu().T, responses.cpu()),
        )
        return fil

    def _laplace_regularized_filter(self, responses, reg_factor):
        # TODO There is an error in the math for the filter computation. It works
        # correctly for square images. However, for non square images only a
        # subset of the image is correctly regularized.
        print(
            "Laplace regularization is not correctly implemented! Only square images are regularized as expected."
        )
        """Compute laplace regularized spike-triggered average

        Args:
            responses (np.array): 2D representation of the response self.report_data.
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
        ti = np.vstack((self.images, np.dot(float(reg_factor), laplace)))
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

    def __init__(self, images, responses, reg_type="ridge regularized", reg_factor=10):
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


class IndividualRegularizationFilter(Filter):
    """Individually regularized linear filter class.

    Class of linear filters with individual regularization factors applied."""

    def __init__(
        self, images, responses, reg_type="ridge regularized", reg_factor=None
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


class Hyperparametersearch:
    """Class of hyperparametersearches of linear filters."""

    def __init__(self, TrainFilter, ValidationFilter, reg_factors, report=True):
        self.TrainFilter = TrainFilter
        self.ValidationFilter = ValidationFilter
        self.reg_factors = reg_factors
        self.report = report
        self.neuron_count = self.TrainFilter.neuron_count
        self.neurons = np.array(list(range(self.neuron_count))).reshape(
            self.neuron_count, 1
        )
        self.time = string_time()

    def _one_hyperparameter(self, reg_factor):
        """Compute linear filter for one regularization factor

        Args:
            reg_factor (float): Regularization factor to use.
        """
        filter = self.TrainFilter.train(reg_factor)
        self.ValidationFilter.predict(filter, True)
        return self.ValidationFilter.single_neuron_correlations

    def conduct_search(self):
        """Conduct hyperparametersearch.

        Returns:
            ndarray: Array of with coloumns: neurons, parameters and correlations.
                Parameters and correlations are 2D arrays themselves.
        """
        self.params = np.empty((self.neuron_count, len(self.reg_factors)))
        self.corrs = np.empty((self.neuron_count, len(self.reg_factors)))
        self.c = np.empty(len(self.reg_factors))
        print("Beginning hyperparametersearch.")
        for counter, reg_factor in track(
            enumerate(self.reg_factors), total=len(self.reg_factors)
        ):
            single_neuron_correlations = self._one_hyperparameter(reg_factor)
            self.corrs[:, counter] = single_neuron_correlations
        for neuron in range(self.neuron_count):
            self.params[neuron, :] = self.reg_factors
        # TODO Fix
        # with ProcessPoolExecutor() as executor:
        #     single_neuron_correlations = list(
        #         track(
        #             executor.map(self._one_hyperparameter, self.reg_factors),
        #             total=len(self.reg_factors),
        #         )
        #     )
        #     for neuron in range(self.neuron_count):
        #         self.params[neuron, :] = self.reg_factors
        #     for counter, value in enumerate(single_neuron_correlations):
        #         self.corrs[:, counter] = value
        print("Concluded hyperparametersearch.")
        self.df_params = pd.DataFrame(self.params, columns=self.reg_factors)
        self.df_corrs = pd.DataFrame(self.corrs, columns=self.reg_factors)
        self.search = np.hstack((self.neurons, self.params, self.corrs))
        return self.search

    def _cmp_best(self, mask):
        """Computes average correlation, single neuron correlation based on
        passed mask.

        Args:
            mask (DataFrame): Boolean DataFrame to use as mask for picking
                hyperparameter
        """
        masked = self.df_params[mask].values.flatten()
        self.hyperparameters = masked[masked == masked.astype(float)].reshape(
            self.neuron_count, 1
        )
        self.single_neuron_correlations = self.df_corrs.max(axis=1).values
        self.single_neuron_correlations = self.single_neuron_correlations.reshape(
            self.neuron_count, 1
        )
        self.results = np.hstack(
            (self.neurons, self.hyperparameters, self.single_neuron_correlations)
        )
        self.avg_correlation = self.single_neuron_correlations.mean()

    def _reporting(self):
        """Reports hyperparametersearch."""
        if self.report:
            np.save(self.report_dir / "hyperparametersearch_report.npy", self.results)
            with open(self.report_dir / "report.md", "w") as f:
                f.write(
                    (
                        "# Readme\n"
                        "hyperparametersearch_report.npy contains a 2D array, "
                        "where column one represents the the neurons, column "
                        "two the regularization factor and column three the "
                        "single neuron correlation of the filter prediction and "
                        "the real responses.\n"
                        "# Average correlation\n"
                        "The best average correlation achieved in this "
                        "hyperparameter search was "
                        f"{round(self.avg_correlation*100,2)} %."
                    )
                )

    def compute_best_parameter(self):
        None

    def get_parameters(self):
        None


class GlobalHyperparametersearch(Hyperparametersearch):
    """Hyperparametersearch for globally regularized linear filters."""

    def __init__(self, TrainFilter, ValidationFilter, reg_factors, report=True):
        super().__init__(TrainFilter, ValidationFilter, reg_factors, report=report)
        self.report_dir = (
            Path.cwd()
            / "reports"
            / "linear_filter"
            / "global_hyperparametersearch"
            / self.time
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _one_hyperparameter(self, reg_factor):
        return super()._one_hyperparameter(reg_factor)

    def conduct_search(self):
        return super().conduct_search()

    def _cmp_best(self, mask):
        return super()._cmp_best(mask)

    def _reporting(self):
        return super()._reporting()

    def compute_best_parameter(self):
        """Pick best regularization factors when globally regularized.

        Returns:
            tuple: Tuple of array [neuron col vector, regularization col vector
                correlation col vector] and average correlation.
        """
        self.hyperparameters = np.empty(self.neuron_count)
        average_corrs = self.df_corrs.mean(axis=0).values
        data = [average_corrs for _ in range(self.neuron_count)]
        self.df_corrs_avg = pd.DataFrame(data, columns=self.reg_factors)
        mask = self.df_corrs_avg.eq(self.df_corrs_avg.max(axis=1), axis=0)
        self._cmp_best(mask)
        self._reporting()
        return self.results, self.avg_correlation

    def get_parameters(self):
        """Get optimized hyperparameters."""
        return self.hyperparameters[0]


class IndividualHyperparametersearch(Hyperparametersearch):
    """Class of hyperparametersearch for single neuron regularized linear filters."""

    def __init__(self, TrainFilter, ValidationFilter, reg_factors, report=True):
        super().__init__(TrainFilter, ValidationFilter, reg_factors, report=report)
        self.report_dir = (
            Path.cwd()
            / "reports"
            / "linear_filter"
            / "individual_hyperparametersearch"
            / self.time
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _one_hyperparameter(self, reg_factor):
        return super()._one_hyperparameter(reg_factor)

    def conduct_search(self):
        return super().conduct_search()

    def _cmp_best(self, mask):
        return super()._cmp_best(mask)

    def _reporting(self):
        return super()._reporting()

    def compute_best_parameter(self):
        """Pick best regularization factors.

        Returns:
            tuple: Tuple of array [neuron col vector, regularization col vector
                correlation col vector] and average correlation.
        """
        self.hyperparameters = np.empty(self.neuron_count)
        mask = self.df_corrs.eq(self.df_corrs.max(axis=1), axis=0)
        self._cmp_best(mask)
        self._reporting()
        return self.results, self.avg_correlation

    def get_parameters(self):
        """Get optimized hyperparameters."""
        return self.hyperparameters


class HyperparameterSearchAnalyzer:
    def __init__(self, report_file, counter):
        (
            cwd,
            reports_dir,
            model_type,
            reg_type,
            date_time,
            report_file,
        ) = report_file.rsplit("/", 5)
        self.report_file_name = report_file
        self.reg_type = reg_type
        self.report_path = Path.cwd() / reports_dir / model_type / reg_type / date_time
        self.model_path = Path.cwd() / "models" / model_type / date_time
        self.report_figures_path = (
            Path.cwd() / reports_dir / "figures" / model_type / reg_type / date_time
        )

        self.report_figures_path.mkdir(parents=True, exist_ok=True)
        self.report_file = self.report_path / report_file
        self.model_file = self.model_path / "evaluated_filter.npy"
        self.counter = counter

    def run(self):
        self.filter = np.load(self.model_file)
        self.report = np.load(self.report_file)
        if self.reg_type == "global_hyperparametersearch":
            self.df_report = pd.DataFrame(
                self.report, columns=["Neuron", "RegFactor", "SingleNeuronCorrelation"]
            )
        else:
            # TODO function for individual_hyperparamsearch
            pass
        if self.report_file_name == "hyperparametersearch_report.npy":
            self.df_report = self.df_report.sort_values(
                "SingleNeuronCorrelation", ascending=False, ignore_index=True
            )
            np.save(
                self.report_path / "hyperparametersearch_report_descending.npy",
                self.df_report.values,
            )

        for i in range(self.counter):
            # pick neuron according to descending order
            neuron = int(self.df_report.Neuron.iloc[i])
            correlation = round(self.df_report.SingleNeuronCorrelation[i] * 100, 2)
            # normalize filter to +-0.5 for visual purposes
            fil = self.filter[neuron, :, :, :]
            fil = torch.from_numpy(fil)
            normer = torchvision.transforms.Normalize(0, 1)
            fil = normer(fil)
            fil = norm(fil)
            fil = fil - 0.5
            fil = fil.squeeze()
            # plot visual representation of filter
            fig, ax = plt.subplots(figsize=figure_sizes["half"])
            im = ax.imshow(fil)
            plt.title(f"Neuron: {neuron} | Correlation: {correlation}")
            fig.colorbar(im, ax=ax, shrink=0.75)
            plt.savefig(
                self.report_figures_path / f"{i:03d}Representation_Neuron_{neuron}.png"
            )
            plt.close()


def load_linear_filter(path: str, device: str = None) -> torch.Tensor:
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = np.load(path)
        data_tensor = torch.from_numpy(data)
        data_tensor.to(device)
    except Exception:
        print("An error occured. Is the file a numpy file? Is path correct?")
    return data_tensor


if __name__ == "__main__":
    pass
