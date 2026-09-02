"""
This module contains the data components for ``armlet``.
"""

from typing import Tuple
import os
import json
import random
import numpy as np
import hydra
import torch

from fluke import DDict

from armlet.data.splitter import DummyDataSplitter
from armlet.data.processing import data_processing_pipeline
from armlet.data.processing.format_conversion import convert_processed_data_to_fluke_data_format
from armlet.data.loading import load_data_from_folder
from armlet.data.saving import save_data
from armlet.data.cleaning import data_cleaning_pipeline
from armlet.data.utils import print_splitted_data_distribution


def data_pipeline(cfg: DDict) -> Tuple[DummyDataSplitter, dict]:
    """Launch the data pipeline before performing the FL training process.

    Args:
        cfg (DDict): configuration dict.

    Returns:
        Tuple[DummyDataSplitter, dict]: the DummyDataSplitter and a dict with val data.
    """

    _set_seed(cfg.data.seed)

    is_static_loading = ("loading" in cfg.data.keys()) and ("static" in cfg.data.loading.keys()) and cfg.data.loading.static
    is_processed_data = is_static_loading and "processing" in cfg.data.loading.load_dir.split("/")[-1]
    is_saving_mode = "saving" in cfg.data.keys()

    if not is_processed_data:

        if is_static_loading:

            splitted_data, is_data_cleaned = load_data_from_folder(cfg.to_dict()["data"], cfg.protocol.n_clients)

        else:

            data = hydra.utils.instantiate(cfg.data.dataset.exclude("dataset_name"))

            data_splitter = hydra.utils.instantiate(
                cfg.data.splitter,
                data_dict=data,
                _recursive_=False,
            )

            splitted_data = data_splitter.assign(cfg.protocol.n_clients)
            is_data_cleaned = False

        #print_splitted_data_distribution(splitted_data)

        if ("cleaning" in cfg.data.keys()) and not is_data_cleaned:

            if is_saving_mode and ("save_data_before_cleaning" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_before_cleaning):
                save_data(splitted_data, cfg.to_dict()["data"], mode="before_cleaning")

            cleaned_data, data_cleaning_metrics = data_cleaning_pipeline(
                data=splitted_data,
                cfg_cleaning=cfg.data.cleaning,
                sensitive_attributes=cfg.data.dataset.sensitive_attributes,
            )

            data_cleaning_metrics_path = os.path.join(cfg.paths.output_dir, "data_cleaning_metrics.json")
            json.dump(data_cleaning_metrics, open(data_cleaning_metrics_path, "w"), indent=4)

            if is_saving_mode and ("save_data_after_cleaning" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_after_cleaning):
                save_data(cleaned_data, cfg.to_dict()["data"], mode="after_cleaning", metrics=data_cleaning_metrics)

        else:
            cleaned_data = splitted_data

        processed_data = data_processing_pipeline(
            data=cleaned_data,
            cfg_data=cfg.data,
        )

        if is_saving_mode and ("save_data_after_processing" in cfg.data.saving.keys()) and (cfg.data.saving.save_data_after_processing):
            save_data(processed_data, cfg.to_dict()["data"], mode="after_processing")

    else:
        processed_data, _ = load_data_from_folder(cfg.to_dict()["data"], cfg.protocol.n_clients)

    data_splitter, additional_data = convert_processed_data_to_fluke_data_format(processed_data, cfg)

    return data_splitter, additional_data

def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

