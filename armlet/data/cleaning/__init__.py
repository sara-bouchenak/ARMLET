"""
This module contains the components for performing data cleaning.
"""

import time
import pandas as pd
import numpy as np
import hydra
from abc import ABC, abstractmethod


def clean_data(data, cfg, sensitive_attributes):
    """Clean the data by following the data cleaning pipeline.

    Warning:
        Label errors and missing values cleaning methods are only applicable to tabular data.
        Missing values cleaning methods can also be used with image datasets.
    """

    all_cleaning_metrics = {}

    start_cleaning_time = time.time()

    for cleaning_cat, cfg_cleaning_cat in cfg.items():
        if cleaning_cat != "name":
            cleaning_method = hydra.utils.instantiate(
                cfg_cleaning_cat,
                sensitive_attributes=sensitive_attributes,
            )
            data, cleaning_metrics = cleaning_method.clean_errors(data)
            all_cleaning_metrics[cleaning_cat] = cleaning_metrics

    end_cleaning_time = time.time()
    cleaning_time = end_cleaning_time - start_cleaning_time
    all_cleaning_metrics["total_cleaning_time"] = cleaning_time

    return data, all_cleaning_metrics

def compute_global_stat_values(clients_data_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]], cfg):

    if "missing_values" in cfg.keys() and cfg.missing_values.name == "global":
        missing_values_correction_mode = cfg.missing_values.correction_mode
    else:
        missing_values_correction_mode = None

    if "outliers" in cfg.keys() and cfg.outliers.name == "global":
        outliers_detection_mode = cfg.outliers.detection_mode
        outliers_correction_mode = cfg.outliers.correction_mode
    else:
        outliers_detection_mode = None
        outliers_correction_mode = None

    global_stat_dict = {"mean": {}, "mode": {}, "std": {}, "q1": {}, "q3": {}}

    X_global = []
    y_global = []
    for id_client, client_tr_data in clients_data_dict.items():
        X, y = client_tr_data
        X_global.append(X)
        y_global.append(y)
    X_global = pd.concat(X_global, axis=0, ignore_index=True)
    y_global = pd.concat(y_global, axis=0, ignore_index=True)

    numeric_cols = X_global.select_dtypes(include=np.number).columns.tolist()
    for col in X_global.columns:
        if col in numeric_cols:
            if outliers_detection_mode == "std":
                global_stat_dict["std"][col] = X_global[col].std()
            elif outliers_detection_mode == "iqr":
                global_stat_dict["q1"][col] = X_global[col].quantile(q=0.25)
                global_stat_dict["q3"][col] = X_global[col].quantile(q=0.75)
            if outliers_detection_mode is not None or missing_values_correction_mode == "mean":
                global_stat_dict["mean"][col] = X_global[col].mean()
            if outliers_correction_mode == "mode" or missing_values_correction_mode == "mode":
                global_stat_dict["mode"][col] = X_global[col].mode()[0]
        else:
            if missing_values_correction_mode is not None:
                global_stat_dict["mode"][col] = X_global[col].mode()[0]

    if missing_values_correction_mode is not None:
        numeric_cols = y_global.select_dtypes(include=np.number).columns.tolist()
        for col in y_global.columns:
            if missing_values_correction_mode == "mean" and col in numeric_cols:
                global_stat_dict["mean"][col] = y_global[col].mean()
            else:
                global_stat_dict["mode"][col] = y_global[col].mode()[0]

    return global_stat_dict


class DataCleaningMethod(ABC):

    def __init__(self, sensitive_attributes: list[str], error_type):
        self.sensitive_attributes = sensitive_attributes
        self.error_type = error_type
        self.n_samples, self.n_detected_errors = self.init_counters()

    def clean_errors(self, all_data):

        start_cleaning_time = time.time()

        all_cleaning_metrics = {}
        cleaned_data = {}

        for id_data, data in all_data.items():

            if "clients" in id_data:
                cleaned_data[id_data], cleaning_metrics = self.clean_client_errors(data)
            else:
                cleaned_data[id_data], cleaning_metrics = self.clean_server_errors(data)

            if cleaning_metrics != {}:
                all_cleaning_metrics[id_data] = cleaning_metrics

        end_cleaning_time = time.time()
        cleaning_time = end_cleaning_time - start_cleaning_time
        all_cleaning_metrics["cleaning_time"] = cleaning_time

        return cleaned_data, all_cleaning_metrics

    def clean_client_errors(self, clients_data_dict):
        self.n_samples, self.n_detected_errors = self.init_counters()
        cleaned_clients_data = {}
        for id_client, client_data in clients_data_dict.items():
            cleaned_clients_data[id_client] = self.clean_errors_dataloader(client_data)
        cleaning_metrics = self.compute_cleaning_metrics()
        return cleaned_clients_data, cleaning_metrics

    def clean_server_errors(self, server_data):
        self.n_samples, self.n_detected_errors = self.init_counters()
        cleaned_server_data = self.clean_errors_dataloader(server_data)
        cleaning_metrics = self.compute_cleaning_metrics()
        return cleaned_server_data, cleaning_metrics

    @abstractmethod
    def clean_errors_dataloader(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def init_counters(self):
        n_samples = {sens_attr: {"group_A": 0, "group_B": 0} for sens_attr in self.sensitive_attributes}
        n_detected_errors = {sens_attr: {"group_A": 0, "group_B": 0} for sens_attr in self.sensitive_attributes}
        return n_samples, n_detected_errors

    def count_n_samples_by_sensitive_attributes(self, X):
        for sens_attr in self.sensitive_attributes:
            n_samples_sa = X.groupby(sens_attr).size()
            n_group_A = n_samples_sa.loc[True] if True in n_samples_sa.index else 0
            self.n_samples[sens_attr]["group_A"] += n_group_A
            n_group_B = n_samples_sa.loc[False] if False in n_samples_sa.index else 0
            self.n_samples[sens_attr]["group_B"] += n_group_B

    def compute_cleaning_metrics(self):
        cleaning_metrics = {}

        tot_n_samples = sum([sum(n_samples_sa.values()) for n_samples_sa in self.n_samples.values()])
        tot_n_detected_errors = sum([sum(n_detected_errors_sa.values()) for n_detected_errors_sa in self.n_detected_errors.values()])
        cleaning_metrics["perc_{}".format(self.error_type)] = tot_n_detected_errors / tot_n_samples if tot_n_samples != 0 else 0

        for sens_attr in self.sensitive_attributes:
            for group in ["group_A", "group_B"]:
                perc_detected_errors = self.n_detected_errors[sens_attr][group] / self.n_samples[sens_attr][group] if self.n_samples[sens_attr][group] != 0 else 0
                cleaning_metrics["perc_{}_{}_{}".format(self.error_type, sens_attr, group)] = perc_detected_errors

        return cleaning_metrics
