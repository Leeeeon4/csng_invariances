import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--report_path", type=str, help="increase output verbosity")
parser.add_argument("--counter", type=int, help="increase output verbosity")
args = parser.parse_args()


def func1(report_path, counter, **kwargs):
    cwd, reports_dir, fil_type, reg_type, date_time, file_name = report_path.rsplit(
        "/", 5
    )
    filter_file = (
        cwd + "/models/" + fil_type + "/" + date_time + "/evaluated_filter.npy"
    )
    eval_fil = np.load(filter_file)
    print(eval_fil.shape)
    data = np.load(report_path)
    df = pd.DataFrame(data, columns=["Neuron", "RegFactor", "Correlation"])
    sorted_df = df
    sorted_df.drop(columns=["Correlation"])
    p, _ = report_path.rsplit("/", 1)
    corrs = pd.read_csv(p + "/Correlations.csv")
    corrs_1 = [float(corrs.columns[1])]
    corrs_1.extend(corrs.iloc[:, 1].to_list())
    sorted_df["Correlation"] = corrs_1
    sorted_df = sorted_df.sort_values(
        ["Correlation"], ascending=False, ignore_index=True
    )
    avg_correlation = sorted_df.Correlation.sum() / len(sorted_df.Correlation)
    with open(p + "/average_correlation.txt", "w") as f:
        f.write(str(avg_correlation))
    sort = sorted_df.values
    # np.save(
    #     p + "/hyperparametersearch_report_descending.npy",
    #     sort,
    # )
    print(sorted_df)
    for i in range(counter):
        corr = sorted_df.Correlation.iloc[i]
        neuron = int(sorted_df.Neuron.iloc[i])
        fil = eval_fil[neuron, 0, :, :]
        print(neuron)
        plt.imshow(fil)
        plt.savefig(f"test_{i}.png")


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


func1(**vars(args))
# com()
