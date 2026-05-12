from typing import Any
import pandas as pd
import os
import math

from print_tables import load_df_results
from utils import filter_and_aggregate_metrics


UTILITY_METRICS = ["accuracy", "precision", "recall", "f1"]
FAIRNESS_METRICS = ["spd", "disp_impact", "disc_index", "eod", "aod"]


def main() -> Any:
    project_dir = "./"
    exp_name = "main_exp"
    exp_dir = os.path.join(project_dir, "traces", exp_name)
    df, metrics_by_cat, other_columns = load_df_results(exp_dir)
    df_agg = filter_and_aggregate_metrics(df, metrics_by_cat, other_columns, agg_func="mean", last_n_rounds=20)

    print_scenarios_sort_cleaning_methods_by_metrics(df_agg)

def print_scenarios_sort_cleaning_methods_by_metrics(df):

    df["cleaning_cat"] = df.apply(cleaning_cat_mapping, axis=1)

    df = add_default_to_each_cleaning_cat(df)

    #df = df[df["dataset"] == "ARS"]
    #df = df[df["model"] == "LogRegression"]

    #df = df.drop(columns=UTILITY_METRICS+["loss"])
    #df = df.drop(columns=["{}_{}".format("age", col) for col in FAIRNESS_METRICS])
    #df = df.drop(columns=["{}_{}".format("race", col) for col in FAIRNESS_METRICS])
    #df = df.drop(columns=["{}_{}".format("gender", col) for col in FAIRNESS_METRICS])

    config_columns = ["dataset", "model", "data_cleaning", "cleaning_cat"]
    df = df.groupby(config_columns, as_index=False).mean(numeric_only=True)

    metrics_dict = {
        "Adult": ["f1", "gender_spd"],
        "ARS": ["f1", "gender_spd"],
        "KDD": ["f1", "gender_spd"],
        "Heart": ["f1", "gender_spd"],
        "MEPS": ["f1", "gender_spd"],
    }

    sort_cleaning_methods_by_metrics(df, metrics_dict, config_columns)

def cleaning_cat_mapping(line):
    if "MV-" in line["data_cleaning"]:
        return "MV"
    elif "OL-" in line["data_cleaning"]:
        return "OL"
    elif "LE-" in line["data_cleaning"]:
        return "LE"
    elif "default" in line["data_cleaning"]:
        return "default"
    else:
        raise

def add_default_to_each_cleaning_cat(df):
    for _, df_dataset in df.groupby("dataset"):
        for cleaning_cat in df_dataset[df_dataset["data_cleaning"] != "default"]["cleaning_cat"].unique():
            df_default = df_dataset[df_dataset["data_cleaning"] == "default"]
            df_default.loc[:, "cleaning_cat"] = cleaning_cat
            df = pd.concat([df, df_default])
    df = df[df["cleaning_cat"] != "default"]
    return df

def sort_cleaning_methods_by_metrics(df, metrics_dict, config_columns):

    group_by_columns = ["dataset", "model", "cleaning_cat"]
    for group_names, df_group in df.groupby(group_by_columns):
        df_group = df_group.drop(columns=group_by_columns)
        df_group = df_group.dropna(axis=1)

        print(" / ".join(["{}: {}".format(column, group_name) for (group_name, column) in zip(group_names, group_by_columns)]))
        print()

        print(df_group.describe().loc[['min', 'max', 'mean', 'std']])
        print()

        metrics = metrics_dict[group_names[0]]

        df_group = add_discretize_metrics(df_group, metrics)

        ranking_metrics = ["{}_interval".format(metric) for metric in metrics]
        sorting_metrics = ranking_metrics + metrics
        df_group = rank_df_by_metrics(df_group, sorting_metrics, ranking_metrics)

        new_config_columns = [col for col in df_group.columns if col in config_columns]
        columns_to_print = ["rank"] + new_config_columns + ranking_metrics + metrics
        print(df_group[columns_to_print].head(30))
        print()

def add_discretize_metrics(df: pd.DataFrame, metrics: list):
    #df_discretized = df_group[metrics].apply(lambda df_col: pd.cut(df_col, bins=5))
    df_discretized = df[metrics].apply(lambda df_col: pd.cut(df_col, bins=range(math.floor(df_col.min()), math.ceil(df_col.max())+1), include_lowest=True))
    discretized_metrics = ["{}_interval".format(metric) for metric in metrics]
    df[discretized_metrics] = df_discretized
    return df

def rank_df_by_metrics(df: pd.DataFrame, sorting_metrics: list, ranking_metrics):
    lower_is_better = [False if metric.replace("_interval", "") in UTILITY_METRICS else True for metric in sorting_metrics]
    df = df.sort_values(by=sorting_metrics, ascending=lower_is_better).reset_index(drop=True)
    df["rank"]= df.groupby(ranking_metrics, sort=False, observed=False).ngroup() + 1
    return df


if __name__ == "__main__":
    main()
