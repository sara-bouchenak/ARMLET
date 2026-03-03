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
