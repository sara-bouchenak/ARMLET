import pandas as pd

from armlet.data.cleaning import DataCleaningMethod


class MissingValuesDataCleaningMethod(DataCleaningMethod):

    def __init__(self, sensitive_attributes: list[str]):
        super().__init__(sensitive_attributes, "missing_values")

    def clean_errors_dataloader(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data == None:
            return None
        else:
            data_cleaned = self.handle_missing_values(data)
            return data_cleaned

    def handle_missing_values(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y = data
        self.count_n_samples_by_sensitive_attributes(X)
        self.count_n_detected_errors_by_sensitive_attributes(data)
        new_data = remove_missing_values(data)
        return new_data

    def count_n_detected_errors_by_sensitive_attributes(self, data: tuple[pd.DataFrame, pd.DataFrame]):
        df = pd.concat(data, axis=1)
        for sens_attr in self.sensitive_attributes:
            n_missing_values_sa = df.isna().any(axis=1).groupby(df[sens_attr]).sum()
            n_group_A = n_missing_values_sa.loc[True] if True in n_missing_values_sa.index else 0
            self.n_detected_errors[sens_attr]["group_A"] += n_group_A
            n_group_B = n_missing_values_sa.loc[False] if False in n_missing_values_sa.index else 0
            self.n_detected_errors[sens_attr]["group_B"] += n_group_B


def remove_missing_values(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat(data, axis=1).dropna().reset_index(drop=True)
    X_cleaned = df.iloc[:, :-1]
    y_cleaned = df.iloc[:, [-1]]
    return (X_cleaned, y_cleaned)
