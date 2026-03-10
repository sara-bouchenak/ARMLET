import pandas as pd


def compute_metrics_name_dict(columns: list):

    other_columns = {}
    other_columns["info"] = []
    if "round" in columns:
        other_columns["info"].append("round")
    if "source" in columns:
        other_columns["info"].append("source")

    all_utility_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss', "training_loss"]
    utility_metrics_list = [column for column in columns if column in all_utility_metrics]

    all_fairness_metrics = ['disp_impact', 'disc_index', 'eod', 'aod', 'spd']
    bias_metrics_dict = {}
    for column in columns:
        if ('_').join(column.split('_')[1:]) in all_fairness_metrics:
            sens_attr = column.split('_')[0]
            if sens_attr not in bias_metrics_dict.keys():
                bias_metrics_dict[sens_attr] = [column]
            else:
                bias_metrics_dict[sens_attr].append(column)

    bias_metrics_list = [bias_metric for sublist in bias_metrics_dict.values() for bias_metric in sublist]

    other_columns["method_pars"] = []
    for column in columns:
        if column not in utility_metrics_list + bias_metrics_list + other_columns["info"]:
            other_columns["method_pars"].append(column)

    metrics_by_cat = bias_metrics_dict
    metrics_by_cat["utility"] = utility_metrics_list
    delete_key = [key for key, val in metrics_by_cat.items() if val == []]
    for key in delete_key:
        del metrics_by_cat[key]

    return metrics_by_cat, other_columns

def preprocess_data(df: pd.DataFrame, metrics_by_cat: dict, other_columns: dict):

    for column in other_columns["method_pars"]:
        df[column] = df[column].astype(str)

    """
    if other_columns["method_pars"] != []:

        cols_without_seed = [col for col in other_columns["method_pars"] if col not in ["exp_seed", "data_seed"]]
        df["method_name"] = df[cols_without_seed].agg(' '.join, axis=1)
        df["method_name"] = df["method_name"].apply(lambda x: x.strip())

        if "exp_seed" in other_columns["method_pars"]:
            cols_with_seed = other_columns["method_pars"]
            df["method_name_with_seed"] = df[cols_with_seed].agg(' '.join, axis=1)
            df["method_name_with_seed"] = df["method_name_with_seed"].apply(lambda x: x.strip())
    """

    for column in other_columns["info"]:
        if column == "round" and column in df.columns:
            df[column] = df[column].astype('int32')

    for cat, metrics in metrics_by_cat.items():
        for metric in metrics:
            if metric in df.columns:
                if "disp_impact" in metric:
                    df[metric] = df[metric].apply(lambda x: (x-1)/(x+1))
                    #df[metric] = df[metric].apply(lambda x: max(x, 1/x) if x != 0 else 10000)
                if "loss" not in metric:
                    df[metric] = df[metric]*100
                if cat != "utility":
                    df[metric] = df[metric].abs()

    return df

def aggregate_metrics_with_mean_of_last_rounds(
    df: pd.DataFrame,
    metrics_by_cat: dict,
    other_columns: dict,
    n_last_rounds: int,
):
    df = df[df["source"] == "server"]
    metrics = [metric for key, sublist in metrics_by_cat.items() for metric in sublist]
    lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds][metrics].mean()
    df_agg = df.groupby(other_columns["method_pars"], as_index=False).apply(lambda_agg)
    return df_agg

def keep_metrics_n_last_rounds(
    df: pd.DataFrame,
    other_columns: dict,
    n_last_rounds: int,
):
    df = df[df["source"] == "server"]
    lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds]
    df_agg = df.groupby(other_columns["method_pars"]).apply(lambda_agg)
    df_agg = df_agg.reset_index(level=other_columns["method_pars"], drop=False)
    return df_agg
