#%%
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm
import time


def test_multiprocessing(args):
    time.sleep(args)


n = range(30)
t = 1


t1 = time.time()

t2 = time.time()

with ProcessPoolExecutor() as executor:
    tqdm(executor.map(test_multiprocessing, [t for _ in n]), total=len(n))

t3 = time.time()

print(f"single process: {round(t2-t1,2)} s\nmultiprocess: {round(t3-t2,2)} s")
#%%
import numpy as np

n = "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-01_10:28:48/hyperparametersearch_report.npy"
a = np.load(n)
a


# %%

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from utility.data_helpers import normalize_tensor_to_0_1 as norm
import torch
import torchvision


figure_sizes = {
    "full": (8, 5.6),
    "half": (5.4, 3.8),
}

mpl.rc_file((str(Path.cwd().parents[0] / "matplotlibrc")))


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
        # # paths in scripts
        # self.report_path = Path.cwd() / reports_dir / model_type / reg_type / date_time
        # self.model_path = Path.cwd() / "models" / model_type / date_time / "evaluated_filter.npy"
        # self.report_figures_path = Path.cwd() / reports_dir / "figures" / model_type / reg_type / date_time

        # paths in ipy usage
        self.report_path = (
            Path.cwd().parents[0] / reports_dir / model_type / reg_type / date_time
        )
        self.model_path = Path.cwd().parents[0] / "models" / model_type / date_time
        self.report_figures_path = (
            Path.cwd().parents[0]
            / reports_dir
            / "figures"
            / model_type
            / reg_type
            / date_time
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


# %%
analyzer = HyperparameterSearchAnalyzer(
    "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-11-01_10:39:26/hyperparametersearch_report.npy",
    50,
)
analyzer.run()
print("Done")

# %%

# %%
