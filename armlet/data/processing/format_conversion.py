import torch
import pandas as pd
import hydra

from fluke.data import FastDataLoader, DummyDataContainer

from armlet.data.splitter import DummyDataSplitter
from armlet.data.processing.audio import make_lazy_audio_loader

def convert_bool_and_cat_to_num(
    subdata,
    dtype='int8',
    **kwargs,
):
    X, y = subdata
    cols = X.select_dtypes(include=["boolean", "category"]).columns.tolist()
    X[cols] = X[cols].astype(dtype)
    y = y.astype(dtype)
    return (X, y)

def convert_dataframes_to_tensors(
        subdata,
        sensitive_attributes=[],
        X_dtype="float32",
        y_dtype="float32",
        **kwargs,
    ):
    X, y = subdata

    ### 1- Move sensitive attributes columns to the end of X
    for sensitive_attribute in sensitive_attributes:
        sensitive_data = X.pop(sensitive_attribute)
        X = pd.concat([X, sensitive_data], axis=1)

    ### 2- Transform X and y to tensors
    X_tensor = torch.tensor(X.values, dtype=getattr(torch, X_dtype))
    y_tensor = torch.tensor(y.values, dtype=getattr(torch, y_dtype))

    return X_tensor, y_tensor

def convert_processed_data_to_fluke_data_format(processed_data, cfg):

    num_classes = _compute_num_classes_from_data(processed_data, cfg)

    if "loader_format" in cfg.data.keys():
        data_loaders = hydra.utils.instantiate(
            {"_target_": cfg.data.loader_format._target_, "_recursive_": False},
            data=processed_data,
            cfg=cfg,
            num_classes=num_classes,
        )
    else:
        data_loaders = convert_tensors_to_fast_data_loaders(
            processed_data, cfg, num_classes
        )

    dummy_data_container = DummyDataContainer(
        data_loaders["clients_train"],
        data_loaders["clients_test"],
        data_loaders["server_test"],
        num_classes,
    )

    data_splitter = DummyDataSplitter(
        dataset=dummy_data_container,
        distribution="",
        client_split=cfg.data.splitter.client_split,
        sampling_perc=cfg.data.others.sampling_perc,
        server_test=cfg.data.splitter.server_test,
        keep_test=cfg.data.splitter.keep_test,
        server_split=cfg.data.splitter.server_split,
        uniform_test=cfg.data.splitter.uniform_test,
    )

    additional_data = {k: v for k, v in data_loaders.items() if k not in 
                       ["clients_train", "clients_test", "server_test"]}

    return data_splitter, additional_data

def _compute_num_classes_from_data(processed_data, cfg):
    if (
        "num_classes" in cfg.data.dataset.keys()
        and cfg.data.dataset.num_classes is not None
    ):
        num_classes = cfg.data.dataset.num_classes
    else:
        unique_classes = []
        for key, data in processed_data.items():
            if data is not None:
                if "clients" in key:
                    for _, sub_data in data.items():
                        if sub_data is not None:
                            unique_classes.append(sub_data[1].flatten().unique())
                else:
                    unique_classes.append(data[1].flatten().unique())
        unique_classes = torch.cat(unique_classes).unique()
        num_classes = len(unique_classes)
    return num_classes

def convert_tensors_to_fast_data_loaders(data, cfg, num_classes):

    fast_data_loaders = {}
    batch_size = cfg.method.hyperparameters.client.batch_size
    sampling_perc = cfg.data.others.sampling_perc

    for data_key in data.keys():

        if "clients_train" in data_key:
            list_dataloaders = []
            for id_client, client_data in data[data_key].items():
                X_tensor, y_tensor = client_data
                dataloader = FastDataLoader(
                    X_tensor,
                    y_tensor,
                    num_labels=num_classes,
                    batch_size=batch_size,
                    shuffle=True,
                    transforms=None,
                    percentage=sampling_perc,
                    skip_singleton=False,
                )
                list_dataloaders.append(dataloader)
            fast_data_loaders[data_key] = list_dataloaders

        if ("clients_test" in data_key) or ("clients_val" in data_key):
            list_dataloaders = []
            for id_client, client_data in data[data_key].items():
                if client_data is not None:
                    if len(client_data) == 3:
                        X_tensor, y_tensor, sa_tensor = client_data
                        dataloader = FastDataLoader(
                            X_tensor,
                            y_tensor,
                            sa_tensor,
                            num_labels=num_classes,
                            batch_size=batch_size,
                            shuffle=False,
                            percentage=sampling_perc,
                            skip_singleton=False,
                        )
                    else:
                        X_tensor, y_tensor = client_data
                        dataloader = FastDataLoader(
                            X_tensor,
                            y_tensor,
                            num_labels=num_classes,
                            batch_size=batch_size,
                            shuffle=False,
                            percentage=sampling_perc,
                            skip_singleton=False,
                        )
                else:
                    dataloader = None
                list_dataloaders.append(dataloader)
            fast_data_loaders[data_key] = list_dataloaders

        if "server_train" in data_key:
            X_tensor, y_tensor = data[data_key]
            fast_data_loaders[data_key] = FastDataLoader(
                X_tensor,
                y_tensor,
                num_labels=num_classes,
                batch_size=batch_size,
                shuffle=True,
                transforms=None,
                percentage=sampling_perc,
                skip_singleton=False,
            )

        if ("server_test" in data_key) or ("server_val" in data_key):
            if data[data_key] != None:
                if len(data[data_key]) == 3:
                    X_tensor, y_tensor, sa_tensor = data[data_key]
                    dataloader = FastDataLoader(
                        X_tensor,
                        y_tensor,
                        sa_tensor,
                        num_labels=num_classes,
                        batch_size=128,
                        shuffle=False,
                        percentage=sampling_perc,
                        skip_singleton=False,
                    )
                else:
                    X_tensor, y_tensor = data[data_key]
                    dataloader = FastDataLoader(
                        X_tensor,
                        y_tensor,
                        num_labels=num_classes,
                        batch_size=128,
                        shuffle=False,
                        percentage=sampling_perc,
                        skip_singleton=False,
                    )
            else:
                dataloader = None
            fast_data_loaders[data_key] = dataloader

    return fast_data_loaders

def convert_lazy_audio_to_data_loaders(data, cfg, num_classes):

    if cfg.data.others.sampling_perc != 1.0:
        raise NotImplementedError(
            "data.others.sampling_perc is not supported for lazy audio loaders yet. "
            "Use sampling_perc=1.0 or add a PyTorch sampler-based lazy sampling path."
        )

    data_loaders = {}
    client_batch_size = cfg.method.hyperparameters.client.batch_size
    server_batch_size = 128
    if "server_batch_size" in cfg.data.loader_format.keys():
        server_batch_size = cfg.data.loader_format.server_batch_size

    for data_key in data.keys():

        if "clients_train" in data_key:
            list_dataloaders = []
            for id_client, client_data in data[data_key].items():
                dataloader = make_lazy_audio_loader(
                    client_data,
                    cfg_loader_format=cfg.data.loader_format,
                    batch_size=client_batch_size,
                    shuffle=True,
                    data_key=f"{data_key}_{id_client}",
                )
                list_dataloaders.append(dataloader)
            data_loaders[data_key] = list_dataloaders

        if ("clients_test" in data_key) or ("clients_val" in data_key):
            list_dataloaders = []
            for id_client, client_data in data[data_key].items():
                if client_data is not None:
                    dataloader = make_lazy_audio_loader(
                        client_data,
                        cfg_loader_format=cfg.data.loader_format,
                        batch_size=client_batch_size,
                        shuffle=False,
                        data_key=f"{data_key}_{id_client}",
                    )
                else:
                    dataloader = None
                list_dataloaders.append(dataloader)
            data_loaders[data_key] = list_dataloaders

        if "server_train" in data_key:
            data_loaders[data_key] = make_lazy_audio_loader(
                    data[data_key],
                    cfg_loader_format=cfg.data.loader_format,
                    batch_size=client_batch_size,
                    shuffle=True,
                    data_key=data_key,
                )

        if ("server_test" in data_key) or ("server_val" in data_key):
            if data[data_key] != None:
                dataloader = make_lazy_audio_loader(
                    data[data_key],
                    cfg_loader_format=cfg.data.loader_format,
                    batch_size=server_batch_size,
                    shuffle=False,
                    data_key=data_key,
                )
            else:
                dataloader = None
            data_loaders[data_key] = dataloader

    return data_loaders
