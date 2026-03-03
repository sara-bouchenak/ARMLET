import pandas as pd
import matplotlib.pyplot as plt

from plot.basics_plot import plot_metrics


def plot_metrics_per_FL_rounds(
        df: pd.DataFrame, 
        metrics_by_cat: dict, 
        method_pars: list[str],
        group_by: list[str] = [],
    ):

    df = df[df["source"] == "server"]
    df = df[df["round"] > 0]

    if group_by != []:
        assert [col in df.columns for col in group_by]
        df = df[df["exp_seed"] == df["exp_seed"].unique()[0]]
        #df = df[df["data_seed"] == df["data_seed"].unique()[0]]

    group_by.extend(["exp_seed", "data_seed"])

    group_cols = [col for col in method_pars if col not in group_by]

    if group_cols == []:
        for metrics_cat, metrics in metrics_by_cat.items():
            if not df[metrics].isna().all().all():
                plot_metrics(
                    df=df,
                    metrics=metrics,
                    plot_type='plot_per_rounds',
                    metrics_cat=metrics_cat,
                    group_by=group_by,
                )

    else:
        for group, df_group in df.groupby(group_cols):
            group_name = ["{}: {}".format(col, val) for col, val in zip(group_cols, group)]
            group_name =  ' / '.join(group_name).strip()

            for metrics_cat, metrics in metrics_by_cat.items():
                if not df_group[metrics].isna().all().all():
                    plot_metrics(
                        df=df_group,
                        metrics=metrics,
                        plot_type='plot_per_rounds',
                        group_name=group_name,
                        metrics_cat=metrics_cat,
                        group_by=group_by,
                    )

    plt.show()

def plot_bar_aggregated_metrics(
    df_agg: pd.DataFrame,
    metrics_by_cat: dict,
    method_pars: list[str],
    x_bar_groups: list[str],
    group_by: list[str] = [],
):

    df_agg = df_agg[df_agg["exp_seed"] == df_agg["exp_seed"].unique()[0]]
    #df = df[df["data_seed"] == df["data_seed"].unique()[0]]

    group_cols = [col for col in method_pars if col not in ["exp_seed", "data_seed"]]
    group_cols = [col for col in group_cols if col not in group_by]
    group_cols = [col for col in group_cols if col not in x_bar_groups]

    if group_cols == []:
        for metrics_cat, metrics in metrics_by_cat.items():
            if not df_agg[metrics].isna().all().all():
                plot_metrics(
                    df=df_agg,
                    metrics=metrics,
                    plot_type='bar',
                    metrics_cat=metrics_cat,
                    x_bar_groups=x_bar_groups,
                    group_by=group_by,
                )

    else:
        for group, df_group in df_agg.groupby(group_cols):
            group_name = ["{}: {}".format(col, val) for col, val in zip(group_cols, group)]
            group_name =  ' / '.join(group_name).strip()

            for metrics_cat, metrics in metrics_by_cat.items():
                if not df_group[metrics].isna().all().all():
                    plot_metrics(
                        df=df_group,
                        metrics=metrics,
                        plot_type='bar',
                        group_name=group_name,
                        metrics_cat=metrics_cat,
                        x_bar_groups=x_bar_groups,
                        group_by=group_by,
                    )

    plt.show()
