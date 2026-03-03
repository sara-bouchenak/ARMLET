"""
This module contains the data components for ``armlet``.
"""

from typing import Tuple
import os
import json
import random
import numpy as np
import hydra

from fluke import DDict
from fluke.data import DummyDataContainer

from armlet.data.splitter import ArmletDataSplitter, DummyDataSplitter
from armlet.data.processing import one_hot_encoding, convert_bool_and_cat_to_num, normalization
from armlet.data.processing import convert_dataframes_to_tensors, load_and_convert_images_to_tensors, convert_tensors_to_fast_data_loaders
from armlet.data.loading import load_data_from_folder
from armlet.data.saving import save_data
from armlet.data.cleaning import clean_data


def data_pipeline(cfg: DDict) -> Tuple[DummyDataSplitter, dict]:
    """Launch the data pipeline before performing the FL training process.

    Args:
        cfg (DDict): configuration dict.

    Returns:
        Tuple[DummyDataSplitter, dict]: the DummyDataSplitter and a dict with val data.
    """

    np.random.seed(cfg.data.seed)
    random.seed(cfg.data.seed)

    is_static_loading = ("loading" in cfg.data.keys()) and ("static" in cfg.data.loading.keys()) and cfg.data.loading.static
    is_tensor_data = is_static_loading and "tensors" in cfg.data.loading.load_dir.split("/")[-1]
    is_saving_mode = "saving" in cfg.data.keys()

    if not is_tensor_data:

        if is_static_loading:

            splitted_data, is_data_cleaned = load_data_from_folder(cfg.to_dict()["data"], cfg.protocol.n_clients)

        else:

            data = hydra.utils.instantiate(cfg.data.dataset.exclude("dataset_name"))

            data_splitter = ArmletDataSplitter(
                data_dict=data,
                dist_cfg=cfg.data.distribution,
                **cfg.data.others,
            )

            splitted_data = data_splitter.assign(cfg.protocol.n_clients)
            is_data_cleaned = False

        if ("cleaning" in cfg.data.keys()) and not is_data_cleaned:

            if is_saving_mode and ("save_data_before_cleaning" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_before_cleaning):
                save_data(splitted_data, cfg.to_dict()["data"], mode="before_cleaning")

            cleaned_data, data_cleaning_metrics = clean_data(splitted_data, cfg=cfg.data.cleaning, sensitive_attributes=cfg.data.dataset.sensitive_attributes)

            data_cleaning_metrics_path = os.path.join(cfg.paths.output_dir, "data_cleaning_metrics.json")
            json.dump(data_cleaning_metrics, open(data_cleaning_metrics_path, "w"), indent=4)

            if is_saving_mode and ("save_data_after_cleaning" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_after_cleaning):
                save_data(cleaned_data, cfg.to_dict()["data"], mode="after_cleaning", metrics=data_cleaning_metrics)

        else:
            cleaned_data = splitted_data

        one_hot_data = one_hot_encoding(cleaned_data)
        numeric_data = convert_bool_and_cat_to_num(one_hot_data)
        normalized_data = normalization(numeric_data, cfg.data.dataset.sensitive_attributes)

        is_image_data = "image_path" in normalized_data["clients_train"]["client_0"][0].columns
        if is_image_data:
            tensor_data = load_and_convert_images_to_tensors(normalized_data, cfg.data.dataset)
        else:
            tensor_data = convert_dataframes_to_tensors(normalized_data, cfg.data.dataset.sensitive_attributes)

        if is_saving_mode and ("save_data_after_conversion_to_tensors" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_after_conversion_to_tensors):
            save_data(tensor_data, cfg.to_dict()["data"], mode="after_tensors")

    else:
        tensor_data, _ = load_data_from_folder(cfg.to_dict()["data"], cfg.protocol.n_clients)

    num_classes = max([len(val[1].squeeze(1).unique()) for val in tensor_data["clients_train"].values()])
    final_data = convert_tensors_to_fast_data_loaders(tensor_data, cfg, num_classes)

    dummy_data_container = DummyDataContainer(
        final_data["clients_train"],
        final_data["clients_test"],
        final_data["server_test"],
        num_classes,
    )

    data_splitter = DummyDataSplitter(
        dataset=dummy_data_container,
        distribution="",
        **cfg.data.others.exclude("client_val_split", "server_test_union", "server_val_split"),
    )

    val_data = {k: v for k, v in final_data.items() if k in ["clients_val", "server_val"]}
    return data_splitter, val_data
