import os
from typing import Any

from load_metrics import load_df_multirun
from utils import compute_metrics_name_dict, preprocess_data, aggregate_metrics_with_mean_of_last_rounds

from plot.specific_plot import plot_metrics_per_FL_rounds, plot_bar_aggregated_metrics


def main() -> Any:
    project_dir = "./"
    exp_name = "main_exp"
    exp_dir = os.path.join(project_dir, "traces", exp_name)

    df, metrics_by_cat, other_columns = load_df_results(exp_dir)

    filter = {
        "dataset": ["Adult", "KDD"],
        "model": ["LogRegression", "SVM"],
        "data_cleaning": ["OL-std-mean-L", "OL-std-mean-G"],
    }
    for key, val in filter.items():
        df = df[df[key].isin(filter[key])]

    group_by = ["model", "data_cleaning"]
    x_bar_groups = ["dataset"]

    plot_metrics_per_FL_rounds(df, metrics_by_cat, other_columns["method_pars"], group_by)

    df_agg = aggregate_metrics_with_mean_of_last_rounds(df, metrics_by_cat, other_columns, n_last_rounds=10)

    plot_bar_aggregated_metrics(df_agg, metrics_by_cat, other_columns["method_pars"], x_bar_groups, group_by)

def load_df_results(exp_dir: str):
    df = load_df_multirun(exp_dir, "perf_global")
    metrics_by_cat, other_columns = compute_metrics_name_dict(df.columns.tolist())
    df = preprocess_data(df, metrics_by_cat, other_columns)
    return df, metrics_by_cat, other_columns


if __name__ == "__main__":
    main()
