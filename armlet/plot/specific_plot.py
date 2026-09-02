import pandas as pd
import matplotlib.pyplot as plt

from armlet.plot.basics_plot import plot_metrics


def plot_metrics_per_FL_rounds(
        df: pd.DataFrame, 
        group_by: list[str] = [],
    ):

    df = df[df["round"] > 0]

    if group_by:
        assert [col in df.columns for col in group_by]
        #df = df[df["exp_seed"] == df["exp_seed"].unique()[0]]
        #df = df[df["data_seed"] == df["data_seed"].unique()[0]]

    group_by.extend(["exp_seed", "data_seed"])

    basic_columns = ["round", "metric_value", "metric_name", "metric_cat", "metric_type"]
    other_columns = [col for col in df.columns if col not in basic_columns]
    group_cols = [col for col in other_columns if col not in group_by]

    if group_cols == []:
        for metric_cat, df_metric_cat in df.groupby("metric_cat"):
            if not df_metric_cat["metric_value"].isna().all().all():
                plot_metrics(
                    df=df_metric_cat,
                    plot_type='plot_per_rounds',
                    metric_cat=str(metric_cat),
                    group_by=group_by,
                )

    else:
        for group, df_group in df.groupby(group_cols):
            group_name = ["{}: {}".format(col, val) for col, val in zip(group_cols, group)]
            group_name =  ' / '.join(group_name).strip()

            for metric_cat, df_metric_cat in df_group.groupby("metric_cat"):
                if not df_metric_cat["metric_value"].isna().all().all():
                    plot_metrics(
                        df=df_metric_cat,
                        plot_type='plot_per_rounds',
                        group_name=group_name,
                        metric_cat=str(metric_cat),
                        group_by=group_by,
                    )

    plt.show()

def plot_bar_aggregated_metrics(
    df_agg: pd.DataFrame,
    x_bar_groups: list[str],
    group_by: list[str] = [],
):

    df_agg = df_agg[df_agg["exp_seed"] == df_agg["exp_seed"].unique()[0]]
    #df = df[df["data_seed"] == df["data_seed"].unique()[0]]

    basic_columns = ["round", "metric_value", "metric_name", "metric_cat", "metric_type"]
    other_columns = [col for col in df_agg.columns if col not in basic_columns]
    group_cols = [col for col in other_columns if col not in ["exp_seed", "data_seed"]]
    group_cols = [col for col in group_cols if col not in group_by]
    group_cols = [col for col in group_cols if col not in x_bar_groups]

    if group_cols == []:
        for metric_cat, df_metric_cat in df_agg.groupby("metric_cat"):
            if not df_metric_cat["metric_value"].isna().all().all():
                plot_metrics(
                    df=df_metric_cat,
                    plot_type='bar',
                    metric_cat=str(metric_cat),
                    x_bar_groups=x_bar_groups,
                    group_by=group_by,
                )

    else:
        for group, df_group in df_agg.groupby(group_cols):
            group_name = ["{}: {}".format(col, val) for col, val in zip(group_cols, group)]
            group_name =  ' / '.join(group_name).strip()

            for metric_cat, df_metric_cat in df_group.groupby("metric_cat"):
                if not df_metric_cat["metric_value"].isna().all().all():
                    plot_metrics(
                        df=df_metric_cat,
                        plot_type='bar',
                        group_name=group_name,
                        metric_cat=str(metric_cat),
                        x_bar_groups=x_bar_groups,
                        group_by=group_by,
                    )

    plt.show()
