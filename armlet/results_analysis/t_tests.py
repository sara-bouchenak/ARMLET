import pandas as pd
import numpy as np
import scipy
import math


def compute_t_tests(df_agg, metrics_by_cat, other_columns, t_test_column, baseline_name):

    varying_params = [col for col in other_columns["method_pars"] if col not in [t_test_column, "exp_seed", "data_seed"]]

    t_test_results = []

    system_info = pd.Series()
    for group_names, df_group in df_agg.groupby(varying_params):
        df_group = df_group.dropna(axis=1)

        for i, param in enumerate(varying_params):
            system_info[param] = group_names[i]

        df_baseline = df_group.loc[df_group[t_test_column] == baseline_name]
        df_baseline = df_baseline.sort_values(by='exp_seed').reset_index()

        for system_name, df_system in df_group.groupby(t_test_column):

            system_info[t_test_column] = system_name

            df_system = df_system.sort_values(by='exp_seed').reset_index()
            assert df_system['exp_seed'].equals(df_baseline['exp_seed'])

            for metrics_cat, metrics in metrics_by_cat.items():
                for metric in metrics:
                    if metric in df_system.columns and "loss" not in metric:

                        t_test_system = system_info.copy()
                        t_test_system["metric"] = metric
                        t_test_system["metric_wo_sa"] = metric.split("{}_".format(metrics_cat))[1] if metrics_cat != "utility" else metric
                        t_test_system["metric_cat"] = metrics_cat if metrics_cat == "utility" else "fairness"
                        t_test_system["metric_cat_sa"] = metrics_cat
                        t_test_system["avg"] = df_system[metric].mean()

                        if system_name != baseline_name:
                            higher_is_better = True if metrics_cat == "utility" else False
                            t_test_system["t-test"] = evaluate_with_t_test(
                                df_baseline[metric].to_numpy(),
                                df_system[metric].to_numpy(),
                                higher_is_better=higher_is_better,
                            )

                        t_test_results.append(t_test_system.to_frame().T)

    df_results = pd.concat(t_test_results, ignore_index=True)
    return df_results

def evaluate_with_t_test(baseline, cleaned, higher_is_better, significant_th=0.05):

    difference = cleaned - baseline
    _, p_value = scipy.stats.ttest_1samp(difference, popmean=0)

    if math.isnan(p_value) or (p_value > significant_th):
        return "insignificant"
    elif (sum(difference) < 0 and higher_is_better) or (sum(difference) > 0 and not higher_is_better):
        return "worse"
    elif (sum(difference) < 0 and not higher_is_better) or (sum(difference) > 0 and higher_is_better):
        return "better"

def aggregate_df_t_tests_with_majority_voting(df_t_tests, other_columns):

    param_columns = [col for col in other_columns["method_pars"] if col not in ["exp_seed", "data_seed"]]
    df_pivot = df_t_tests.pivot(index=param_columns, columns=["metric_cat", "metric_cat_sa", "metric"], values="t-test")

    df_pivot = df_pivot.T.groupby(level=[0, 1]).apply(compute_majority_voting).T

    df_pivot = df_pivot.melt(col_level=0, id_vars=["utility"], ignore_index=False)
    df_pivot = df_pivot.drop("metric_cat", axis=1)
    df_pivot = df_pivot.dropna()
    df_pivot = df_pivot.rename(columns={"value": "fairness"})

    df_pivot = df_pivot[["utility", "fairness"]].value_counts(sort=False)
    df_pivot = df_pivot.div(df_pivot.sum()).mul(100).round(0)
    return df_pivot

def compute_majority_voting(df_metrics_cat_sa):

    def classify_vote(col):
        if col == 0:
            return "insignificant"
        elif col < 0:
            return "worse"
        elif col > 0:
            return "better"
        else:
            return np.nan

    df_metrics_cat_sa = df_metrics_cat_sa.replace("insignificant", 0)
    df_metrics_cat_sa = df_metrics_cat_sa.replace("worse", -1)
    df_metrics_cat_sa = df_metrics_cat_sa.replace("better", 1)
    df_metrics_cat_sa = df_metrics_cat_sa.sum(skipna=False)
    df_metrics_cat_sa = df_metrics_cat_sa.apply(classify_vote)
    return df_metrics_cat_sa

def aggregate_df_t_tests_by(df_t_tests, by):
    df_pivot = df_t_tests.pivot_table(
        index=by,
        columns=["metric_cat", "t-test"],
        values="avg",
        aggfunc="count",
        fill_value=0.0,
    )
    df_pivot = df_pivot.T.groupby(level=0, group_keys=False).apply(
        lambda df_group: df_group.div(df_group.sum(axis=0), axis=1).mul(100).round(0)
    ).T
    return df_pivot
