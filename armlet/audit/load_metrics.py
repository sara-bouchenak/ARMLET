import os
import json
import pandas as pd


UTILITY_METRICS = ["accuracy", "precision", "recall", "f1", "loss", "training_loss", "roc_auc"]
FAIRNESS_METRICS = ["spd", "disp_impact", "disc_index", "eod", "aod"]


def load_df_multirun(exp_dir: str, metric_types: list[str]):

    json_metrics_paths = []
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith("results.json"):
                json_metrics_paths.append(os.path.join(root, file))

    df = []
    for path in json_metrics_paths:
        df_run = load_df_run(path, metric_types)
        pars = path.split(exp_dir)[1].replace(os.path.sep, ",").split(",")[1:-1]
        pars = {par.split("=")[0]: par.split("=")[1] for par in pars}
        for key, val in pars.items():
                df_run[key] = val
        df.append(df_run)
    df = pd.concat(df, axis=0, ignore_index=True)
    return df

def load_df_run(json_metrics_path: str, metric_types: list[str]):

    with open(json_metrics_path, "r") as f:
            json_metrics = json.load(f)

    lines = []
    columns = ["metric_type", "metric_name", "metric_value", "source", "round"]

    for metric_type in metric_types:
        if metric_type in json_metrics.keys():
            metrics_dict = json_metrics[metric_type]

            if metric_type == "perf_global":
                source = "server"
                for round, sub_metrics_dict in metrics_dict.items():
                    for metric_name, metric_value in sub_metrics_dict.items():
                        line = [metric_type, metric_name, metric_value, source, round]
                        lines.append(line)

            else:
                for round, sub_metrics_dict in metrics_dict.items():
                    for source, metrics in sub_metrics_dict.items():
                        source = "client_{}".format(source) if source != "server" else source
                        for metric_name, metric_value in metrics.items():
                            line = [metric_type, metric_name, metric_value, source, round]
                            lines.append(line)

    df_metrics = pd.DataFrame(data=lines, columns=columns)
    #df_metrics["metric_type"] = df_metrics["metric_type"].replace("perf_", "")
    return df_metrics

def preprocess_df_metrics(df: pd.DataFrame, no_metric_cat=False):

    # Convert round values
    df["round"] = df["round"].astype('int32')

    # Convert run parameters to string
    base_columns = ["metric_type", "metric_name", "metric_value", "source", "round"]
    run_pars = [col for col in df.columns if col not in base_columns]
    for column in run_pars:
        df[column] = df[column].astype(str)

    #compute total model_size_in_bits and n_model_params
    for metric in ["model_size_in_bits", "n_model_params"]:
        if metric in df["metric_name"].unique():
            df = compute_tot_comm_costs(df, metric)

    def categorize_metrics(metric_name):
        if metric_name in UTILITY_METRICS:
            return "utility"
        elif ('_').join(metric_name.split('_')[1:]) in FAIRNESS_METRICS:
            return "fairness"
        else:
            return "other"
    df["metric_cat"] = df["metric_name"].apply(categorize_metrics)

    # process disparate impact values
    mask_disp_impact = df["metric_name"].apply(lambda x: "disp_impact" in x)
    df.loc[mask_disp_impact, "metric_value"] = df.loc[mask_disp_impact, "metric_value"].apply(
        lambda x: (x-1)/(x+1))

    # convert utility and fairness metrics (but not loss) to percentages
    mask_perc_columns = df.apply(
        lambda x: x["metric_cat"] in ["utility", "fairness"] and "loss" not in x["metric_name"],
        axis=1)
    df.loc[mask_perc_columns, "metric_value"] = df.loc[mask_perc_columns, "metric_value"] * 100

    # Keep absolute values of fairness metrics
    mask_fairness = df["metric_cat"] == "fairness"
    df.loc[mask_fairness, "metric_value"] = df.loc[mask_fairness, "metric_value"].abs()

    if no_metric_cat:
        df["metric_cat"] = "unknown"

    return df

def compute_tot_comm_costs(df, metric):
    df_metric = df[df["metric_name"] == metric]
    group_by_col = [col for col in df.columns if col not in
                    ["source", "metric_type", "metric_value", "metric_name"]]
    df_tot = df_metric.groupby(group_by_col).aggregate({"metric_value": sum})
    df_tot = df_tot.reset_index(level=group_by_col, drop=False)
    df_tot["metric_name"] = "tot_{}".format(metric)
    df_tot["source"] = "general"
    df_tot["metric_type"] = "comm_costs"
    df = pd.concat([df, df_tot])
    return df

def filter_and_aggregate_metrics(
    df: pd.DataFrame,
    group_by_columns: list[str],
    last_n_rounds: int | None = 1,
    agg_func: str | None = None,
):

    if last_n_rounds is not None:
        lambda_filter = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - last_n_rounds]
        df_filter = df.groupby(group_by_columns).apply(lambda_filter)
        df_filter = df_filter.reset_index(level=group_by_columns, drop=False)
    else:
        df_filter = df

    if agg_func is not None:
        lambda_agg = lambda df_lambda: df_lambda["metric_value"].aggregate(agg_func)
        df_agg = df_filter.groupby(group_by_columns).apply(lambda_agg)
        df_agg = df_agg.reset_index(level=group_by_columns, drop=False)
        df_agg = df_agg.rename(columns={0: "metric_value"})
    else:
        df_agg = df_filter

    return df_agg
