import torch
import pandas as pd

from fluke.data import FastDataLoader, DummyDataContainer

from armlet.data.splitter import DummyDataSplitter


def convert_bool_and_cat_to_num(subdata, data_key):
    X, y = subdata
    cols = X.select_dtypes(include=["boolean", "category"]).columns.tolist()
    X[cols] = X[cols].astype('int8')
    y = y.astype('int8')
    return (X, y)

def convert_dataframes_to_tensors(subdata, data_key, sensitive_attributes):
    ### 1- Move sensitive attributes columns to the end of X
    ### 2- Transform X and y to tensors

    X, y = subdata
    for sensitive_attribute in sensitive_attributes:
        sensitive_data = X.pop(sensitive_attribute)
        X = pd.concat([X, sensitive_data], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return X_tensor, y_tensor

def convert_tensors_to_fluke_data_format(tensor_data, cfg):

    num_classes = max([len(val[1].squeeze(1).unique()) for val in tensor_data["clients_train"].values()])
    fast_data_loaders = convert_tensors_to_fast_data_loaders(tensor_data, cfg, num_classes)

    dummy_data_container = DummyDataContainer(
        fast_data_loaders["clients_train"],
        fast_data_loaders["clients_test"],
        fast_data_loaders["server_test"],
        num_classes,
    )

    data_splitter = DummyDataSplitter(
        dataset=dummy_data_container,
        distribution="",
        **cfg.data.others.exclude("client_val_split", "server_test_union", "server_val_split"),
    )

    val_data = {k: v for k, v in fast_data_loaders.items() if k in ["clients_val", "server_val"]}

    return data_splitter, val_data

def convert_tensors_to_fast_data_loaders(data, cfg, num_classes):

    fast_data_loaders = {}
    batch_size = cfg.method.hyperparameters.client.batch_size
    sampling_perc = cfg.data.others.sampling_perc

    list_dataloaders = []
    for id_client, client_data in data["clients_train"].items():
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
    fast_data_loaders["clients_train"] = list_dataloaders

    for key in ["clients_test", "clients_val"]:
        list_dataloaders = []
        for id_client, client_data in data[key].items():
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
        fast_data_loaders[key] = list_dataloaders

    for key in ["server_test", "server_val"]:
        if data[key] != None:
            if len(data[key]) == 3:
                X_tensor, y_tensor, sa_tensor = data[key]
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
                X_tensor, y_tensor = data[key]
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
        fast_data_loaders[key] = dataloader

    return fast_data_loaders
