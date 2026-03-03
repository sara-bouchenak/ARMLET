import os
import json
import pandas as pd


def load_df_multirun(exp_dir: str, metrics_type: str):

    json_metrics_paths = []
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith("results.json"):
                json_metrics_paths.append(os.path.join(root, file))

    df = []
    for path in json_metrics_paths:
        df_run = load_df(path, metrics_type)
        pars = path.split(exp_dir)[1].replace(os.path.sep, ",").split(",")[1:-1]
        pars = {par.split("=")[0]: par.split("=")[1] for par in pars}
        for key, val in pars.items():
                df_run[key] = val
        df.append(df_run)
    df = pd.concat(df, axis=0, ignore_index=True)
    return df

def load_df(json_metrics_path: str, metrics_type: str):

    with open(json_metrics_path, "r") as f:
            metrics_dict = json.load(f)

    if metrics_type == "perf_global":
        metrics_subdict = metrics_dict[metrics_type]
        df = compute_perf_global_metrics(metrics_subdict)
    elif metrics_type == "perf_locals":
        metrics_subdict = metrics_dict[metrics_type]
        df = compute_perf_locals_metrics(metrics_subdict)
    elif metrics_type == "perf_prefit" or metrics_type == "perf_postfit":
        metrics_subdict = metrics_dict[metrics_type]
        df = compute_perf_prefit_postfit_metrics(metrics_subdict)
    else:
        raise TypeError("metrics_type '{}' is unknown!".format(metrics_type))
    return df

def compute_perf_global_metrics(perf_global_dict: dict):
    lines = []
    columns = []
    for round, metrics in perf_global_dict.items():
        if columns == []:
            columns = list(metrics.keys())
        line = [metric for metric in metrics.values()]
        line.append(round)
        lines.append(line)
    columns.append("round")
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    df_metrics["source"] = "server"
    return df_metrics

def compute_perf_locals_metrics(perf_local_dict: dict):
    lines = []
    columns = []
    for round, clients_dict in perf_local_dict.items():
        for id_client, metrics in clients_dict['null'].items():
            if columns == []:
                columns = list(metrics.keys())
            line = [metric for metric in metrics.values()]
            line.append(id_client)
            line.append(round)
            lines.append(line)
    columns.extend(["id_client", "round"])
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    return df_metrics

def compute_perf_prefit_postfit_metrics(perf_dict: dict):
    lines = []
    columns = []
    for round, clients_dict in perf_dict.items():
        for id_client, metrics in clients_dict.items():
            if columns == []:
                columns = list(metrics.keys())
            line = [metric for metric in metrics.values()]
            line.append(id_client)
            line.append(round)
            lines.append(line)
    columns.extend(["id_client", "round"])
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    return df_metrics
