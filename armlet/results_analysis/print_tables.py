import os
from typing import Any

from load_metrics import load_df_multirun
from utils import compute_metrics_name_dict, preprocess_data
from utils import aggregate_metrics_with_mean_of_last_rounds, keep_metrics_n_last_rounds
from t_tests import compute_t_tests, aggregate_df_t_tests_for_joint_impact, aggregate_df_t_tests_by


def main() -> Any:
    project_dir = "./"
    exp_name = "main_exp"
    exp_dir = os.path.join(project_dir, "traces", exp_name)

    df, metrics_by_cat, other_columns = load_df_results(exp_dir)
    
    #df_agg = aggregate_metrics_with_mean_of_last_rounds(df, metrics_by_cat, other_columns, n_last_rounds=20)
    df_agg = keep_metrics_n_last_rounds(df, other_columns, n_last_rounds=20)

    df_t_tests = compute_t_tests(
        df_agg,
        metrics_by_cat,
        other_columns,
        t_test_column="data_cleaning",
        baseline_name="default",
        significant_th=0.01,
    )

    df_joint = aggregate_df_t_tests_for_joint_impact(df_t_tests, metrics_by_cat, other_columns)
    print(df_joint)
    print()

    df_metrics_pivot = aggregate_df_t_tests_by(df_t_tests, by="metric")
    print(df_metrics_pivot)
    print()

    df_metrics_wo_sa_pivot = aggregate_df_t_tests_by(df_t_tests, by="metric_wo_sa")
    print(df_metrics_wo_sa_pivot)
    print()

    df_cleaning_pivot = aggregate_df_t_tests_by(df_t_tests, by="data_cleaning")
    print(df_cleaning_pivot)
    print()

    df_dataset_pivot = aggregate_df_t_tests_by(df_t_tests, by="dataset")
    print(df_dataset_pivot)
    print()

    df_model_pivot = aggregate_df_t_tests_by(df_t_tests, by="model")
    print(df_model_pivot)
    print()

def load_df_results(exp_dir: str):
    df = load_df_multirun(exp_dir, "perf_global")
    metrics_by_cat, other_columns = compute_metrics_name_dict(df.columns.tolist())
    df = preprocess_data(df, metrics_by_cat, other_columns)
    return df, metrics_by_cat, other_columns


if __name__ == "__main__":
    main()
