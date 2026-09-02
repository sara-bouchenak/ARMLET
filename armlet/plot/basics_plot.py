import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from armlet.plot.save import save_plot_to_png


METRIC_TO_LABEL = {
    'accuracy': "Accuracy (%)",
    'precision': "Precision (%)",
    'recall': "Recall (%)",
    'f1': "F1-score (%)",
    'loss': "Loss",
    'spd': "Statistical \n Parity \n Difference (%)",
    'disp_impact': "Disparate \n Impact (%)",
    'disc_index': "Discrimination \n Index (%)",
    'eod': "Equal \n Opportunity \n Difference (%)",
    'aod': "Average \n Odds \n Difference (%)",
}

COLORS = ['#A8D8B9', '#87CEEB', '#B47ECF', '#ECD266', '#F4A460', '#FF6347',
          "#2267FC", "#8906AA", "#02AA2C", "#C40303", "#35C8E2", "#FA70EE"]

MARKERS = ['o', '^', 's', 'x', 'd', 'P', 'h', 'v', 'p', '*', '<', '>']


def plot_metrics(
    df: pd.DataFrame,
    plot_type: str,
    group_name: str = "",
    metric_cat: str = "",
    group_by: list[str] = [],
    x_bar_groups: list[str] = [],
    save_plot: bool = False,
    save_plot_dir: str = "",
):

    plt.rcParams.update({'pdf.fonttype': 42})
    figsize = (18, 12)
    fig, axes = plt.subplots(df["metric_name"].nunique(), figsize=figsize, sharex=True)

    title = group_name
    if metric_cat != "":
        if title != "":
            title += " - "
        if metric_cat == "utility":
            title += "Utility metrics \n"
        elif metric_cat == "data_selection":
            title += "Data selection metrics \n"
        elif metric_cat == "custom_fields":
            title += "Other metrics \n"
        else:
            title += "Fairness metrics {}".format(metric_cat)
        fig.suptitle(title, fontsize=14)

    for i, (metric, df_metric) in enumerate(df.groupby("metric_name")):
        ax = axes[i] if df["metric_name"].nunique() >= 2 else axes
        if plot_type == 'bar':
            plot_bar_metric(df_metric, metric, ax, i, metric_cat, group_by, x_bar_groups)
        elif plot_type == 'plot_per_rounds':
            plot_metric_per_rounds(df_metric, metric, ax, i, metric_cat, group_by)
        else:
            raise TypeError("plot_type '{}' is unknown!".format(plot_type)) 

    plt.tight_layout()

    if save_plot:
        plot_name = metric_cat if metric_cat != 0 else df["metric_name"].unique()[0]
        save_plot_to_png(fig, plot_type, plot_name, save_plot_dir)

def plot_bar_metric(df, metric, ax, i, metrics_cat, group_by, x_bar_groups):

    if group_by:
        assert [col in df.columns for col in group_by] 
    assert [col in df.columns for col in x_bar_groups]

    df_cleaned = df.dropna(subset=["metric_value"])

    df_cleaned["x_bar_groups"] = df_cleaned[x_bar_groups].agg(' '.join, axis=1)
    df_cleaned["x_bar_groups"] = df_cleaned["x_bar_groups"].apply(lambda x: x.strip())

    metric_name = metric.split(metrics_cat+"_")[1] if (metrics_cat != "" and metrics_cat in metric) else metric
    metric_label = METRIC_TO_LABEL[metric_name] if metric_name in METRIC_TO_LABEL else metric_name

    if group_by:
        df_cleaned["group_by"] = df_cleaned[group_by].agg(' '.join, axis=1)
        df_cleaned["group_by"] = df_cleaned["group_by"].apply(lambda x: x.strip())
        n_groups = df_cleaned["group_by"].nunique()
        df_pivot = df_cleaned.pivot(index="x_bar_groups", columns="group_by", values="metric_value")
        df_pivot.plot.bar(ax=ax, ylabel=metric_label, color=COLORS[:n_groups])
    else:
        df_cleaned.plot.bar(ax=ax, x="x_bar_groups", y="metric_value", ylabel=metric_label, color=COLORS[0])

    #desired_order = df[y_column].unique().tolist()
    #first_element = desired_order.pop(-1)
    #desired_order.insert(0, first_element)
    #df_pivot = df_pivot.reindex(desired_order)

    if i == 0 and group_by:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()
    ax.grid(True)
    ax.set_xlabel(' / '.join(x_bar_groups).strip())
    ax.tick_params("x", rotation=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    """
    if metric == "accuracy":
        ax.set_ylim(60, 85)
    elif metric == "precision":
        ax.set_ylim(40, 80)
    elif metric == "recall":
        ax.set_ylim(15, 65)
    elif metric == "f1":
        ax.set_ylim(25, 70)
    """

def plot_metric_per_rounds(df, metric, ax, i, metrics_cat, group_by):

    assert "round" in df.columns
    assert [col in df.columns for col in group_by]

    metric_name = metric.split(metrics_cat+"_")[1] if (metrics_cat != "" and metrics_cat in metric) else metric
    metric_label = METRIC_TO_LABEL[metric_name] if metric_name in METRIC_TO_LABEL else metric_name

    groups = [group for group, _ in df.groupby(group_by)]
    groups = [["{}: {}".format(col, val) for col, val in zip(group_by, group)] for group in groups]
    groups = [' / '.join(group).strip() for group in groups]

    groups_to_color = {group: COLORS[i] for i, group in enumerate(groups)}
    groups_to_marker = {group: MARKERS[i] for i, group in enumerate(groups)}

    for group, df_group in df.groupby(group_by):
        group = ["{}: {}".format(col, val) for col, val in zip(group_by, group)]
        group =  ' / '.join(group).strip()

        df_group.plot(x="round", y="metric_value", ax=ax, ylabel=metric_label, xlabel="FL rounds",
                      label=group, color=groups_to_color[group], marker=groups_to_marker[group],
                      markevery=20, markerfacecolor="None")

    if i == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()
    ax.grid(True)

    #if metric == "accuracy":
    #    ax.set_ylim(0, 100)
