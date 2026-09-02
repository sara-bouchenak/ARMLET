import os
from typing import Any

from armlet.audit.load_metrics import load_df_multirun, preprocess_df_metrics, filter_and_aggregate_metrics
from armlet.plot.specific_plot import plot_metrics_per_FL_rounds, plot_bar_aggregated_metrics


def main() -> Any:
    project_dir = "./"
    exp_name = "example/adult_benchmark"
    exp_dir = os.path.join(project_dir, "outputs", exp_name)

    metric_types = ["perf_global", "perf_locals", "perf_prefit", "perf_postfit"]
    df = load_df_multirun(exp_dir, metric_types)
    df = preprocess_df_metrics(df)

    df = df.loc[df["metric_type"] == "perf_global", :]

    #filter = {
    #    "dataset": ["Adult", "KDD"],
    #    "model": ["LogRegression", "SVM"],
    #    "data_cleaning": ["OL-std-mean-L", "OL-std-mean-G"],
    #}
    #for key, val in filter.items():
    #    df = df[df[key].isin(val)]

    group_by = ["dataset"]
    x_bar_groups = ["model"]

    plot_metrics_per_FL_rounds(df, group_by)

    group_by_columns = [col for col in df.columns if col not in ["round", "metric_value"]]
    df_agg = filter_and_aggregate_metrics(df, group_by_columns, agg_func="mean", last_n_rounds=10)

    plot_bar_aggregated_metrics(df_agg, x_bar_groups, group_by)


if __name__ == "__main__":
    main()
