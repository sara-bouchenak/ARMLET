import os
from typing import Any

from load_metrics import load_df_multirun
from utils import compute_metrics_name_dict, preprocess_data, aggregate_metrics_with_mean_of_last_rounds
from t_tests import compute_t_tests, aggregate_df_t_tests_with_majority_voting, aggregate_df_t_tests_by


def main() -> Any:
    project_dir = "./"
    exp_name = "main_exp"
    exp_dir = os.path.join(project_dir, "traces", exp_name)

    df, metrics_by_cat, other_columns = load_df_results(exp_dir)

    df_agg = aggregate_metrics_with_mean_of_last_rounds(df, metrics_by_cat, other_columns, n_last_rounds=10)

    """
    df_agg = df_agg.drop(columns=["data_seed"])
    tables_dir = os.path.join(project_dir, "outputs", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    table_path = "{}/{}_agg_table.csv".format(tables_dir, exp_name)
    df_agg.to_csv(table_path)
    """

    df_t_tests = compute_t_tests(df_agg, metrics_by_cat, other_columns, t_test_column="data_cleaning", baseline_name="default")

    df_global = aggregate_df_t_tests_with_majority_voting(df_t_tests, other_columns)
    print()
    print(df_global)
    print()

    df_metrics_pivot = aggregate_df_t_tests_by(df_t_tests, by="metric_wo_sa")
    print(df_metrics_pivot)
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
