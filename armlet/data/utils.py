"""
This module contains the general data utilities.
"""

import pandas as pd
import warnings

from sklearn.model_selection import train_test_split


def dataframe_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
):
    if test_size == 0.0:
        return X, None, y, None
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, train_size=train_size, random_state=random_state
        )
        X_tr = X_tr.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        X_te = X_te.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)
        return X_tr, X_te, y_tr, y_te

def dataframe_safe_train_test_split(
    X: pd.DataFrame, y: pd.DataFrame, test_size: float, client_id: int | None = None
):
    if test_size == 0.0:
            return X, None, y, None
    else:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y)
        except ValueError:
            client_str = f"[Client {client_id}]" if client_id is not None else ""
            warnings.warn(f"Stratified split failed for {client_str}. Falling back to random split.")
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size)
        X_tr = X_tr.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        X_te = X_te.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)
        return X_tr, X_te, y_tr, y_te

def print_splitted_data_distribution(splitted_data):
    df_dist = []
    for key, data in splitted_data.items():
        if "clients" in key:
            for sub_key, sub_data in data.items():
                df_dist.append([key, sub_key, sub_data[1].shape[0]])
        else:
            df_dist.append([key, "server", data[1].shape[0]])
    df_dist = pd.DataFrame(df_dist, columns=["split", "source", "n_samples"])
    df_dist["perc_samples"] = df_dist.groupby("split", group_keys=False).apply(
        lambda df_group: df_group["n_samples"] / df_group["n_samples"].sum())
    df_dist = df_dist.sort_values("source")
    print(df_dist)
