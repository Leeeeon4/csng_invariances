#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

# parser = argparse.ArgumentParser()
# parser.add_argument("--report_file", type=str, help="increase output verbosity")
# parser.add_argument("--counter", type=int, help="increase output verbosity")
# args = parser.parse_args()
report_file = "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-10-26_17:31:40/hyperparametersearch_report.npy"
counter = 5


def func1(report_file, counter, **kwargs):
    cwd, reports_dir, fil_type, reg_type, date_time, file_name = report_file.rsplit(
        "/", 5
    )
    report_figure_path = (
        Path.cwd().parents[1]
        / reports_dir
        / "figures"
        / fil_type
        / reg_type
        / date_time
    )
    report_figure_path.mkdir(parents=True, exist_ok=True)
    filter_file = (
        Path.cwd().parents[1] / "models" / fil_type / date_time / "evaluated_filter.npy"
    )
    data = np.load(report_file)
    df = pd.DataFrame(data, columns=["Neuron", "RegFactor", "Correlation"])
    sorted_df = df
    sorted_df.drop(columns=["Correlation"])
    report_path, _ = report_file.rsplit("/", 1)
    corrs = pd.read_csv(report_path + "/Correlations.csv")
    corrs_1 = [float(corrs.columns[1])]
    corrs_1.extend(corrs.iloc[:, 1].to_list())
    sorted_df["Correlation"] = corrs_1
    sorted_df = sorted_df.sort_values(
        ["Correlation"], ascending=False, ignore_index=True
    )
    avg_correlation = sorted_df.Correlation.sum() / len(sorted_df.Correlation)
    with open(report_path + "/average_correlation.txt", "w") as f:
        f.write(str(avg_correlation))
    sort = sorted_df.values
    np.save(
        report_path + "/hyperparametersearch_report_descending.npy",
        sort,
    )
    print(sorted_df)
    return sorted_df, filter_file


def func2(counter, sorted_df, filter_file):
    eval_fil = np.load(filter_file)
    for i in range(counter):
        corr = sorted_df.Correlation.iloc[i]
        neuron = int(sorted_df.Neuron.iloc[i])
        fil = eval_fil[neuron, 0, :, :]
        print(neuron)
        plt.imshow(fil)
        plt.show()


def com():
    ind_path = "/home/leon/csng_invariances/reports/linear_filter/individual_hyperparametersearch/2021-10-14_18:15:45/Correlations.csv"
    glob_path = "/home/leon/csng_invariances/reports/linear_filter/global_hyperparametersearch/2021-10-14_18:13:10/Correlations.csv"
    glob = pd.read_csv(glob_path)
    ind = pd.read_csv(ind_path)
    glob = glob.iloc[:, 1]
    ind = ind.iloc[:, 1]
    delta = ind - glob
    for i in delta:
        if i < -1e-15:
            print("something went wrong")
    print("done")


com()
#%%
a, b = func1(report_file, counter)

# %%
func2(counter, a, b)
# %%
