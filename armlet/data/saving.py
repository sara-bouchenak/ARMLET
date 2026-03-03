"""
This module contains the functions for saving data at a given step
of the data pipeline.
"""

import os
import yaml
import json
import torch


def save_data(
    data,
    data_cfg_dict: dict,
    mode: str,
    metrics: dict = {},
):

    save_dir = data_cfg_dict["saving"]["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if mode == "before_cleaning":
        subdir = "no_cleaning"
        data_cfg_dict["cleaning"] = None
    elif mode == "after_cleaning":
        subdir = "{}_cleaning".format(data_cfg_dict["cleaning"]["name"])
    else:
        subdir = mode

    save_subdir = os.path.join(save_dir, subdir)
    if os.path.exists(save_subdir):
        print("Subdir \"{}\" already exists! Data is not saved!".format(save_subdir))
        return 0
    else:
        os.makedirs(save_subdir)

    for id_client, client_tr in data["clients_train"].items():
        client_path = os.path.join(save_subdir, "{}".format(id_client))
        os.mkdir(client_path)
        if mode == "after_tensors":
            save_tensors_to_pt(client_tr, client_path, id_client, "tr")
        else:
            save_data_to_pkl(client_tr, client_path, id_client, "tr")

        for (data_key, data_type) in [("clients_test", "te"), ("clients_val", "val")]:
            client_data = data[data_key][id_client]
            if client_data is not None:
                if mode == "after_tensors":
                    save_tensors_to_pt(client_data, client_path, id_client, data_type)
                else:
                    save_data_to_pkl(client_data, client_path, id_client, data_type)

    server_path = os.path.join(save_subdir, "server")
    os.mkdir(server_path)

    for (data_key, data_type) in [("server_test", "te"), ("server_val", "val")]:
        if data[data_key] is not None:
            if mode == "after_tensors":
                save_tensors_to_pt(data[data_key], server_path, "server", data_type)
            else:
                save_data_to_pkl(data[data_key], server_path, "server", data_type)

    data_cfg_dict["n_clients"] = len(data["clients_train"])

    data_config_path = os.path.join(save_subdir, "data_config.yaml")
    yaml.dump(data_cfg_dict, open(data_config_path, "w"))

    if len(metrics) > 0:
        data_cleaning_metrics_path = os.path.join(save_subdir, "data_cleaning_metrics.json")
        json.dump(metrics, open(data_cleaning_metrics_path, "w"), indent=4)

def save_data_to_pkl(data, path: str, id: str, data_type: str):
    X, y = data
    X_path = os.path.join(path, "{}_X_{}.pkl".format(id, data_type))
    X.to_pickle(X_path)
    y_path = os.path.join(path, "{}_y_{}.pkl".format(id, data_type))
    y.to_pickle(y_path)

def save_tensors_to_pt(data, path: str, id: str, data_type: str):
    if len(data) == 3:
        X, y, sa = data
        sa_path = os.path.join(path, "{}_sa_{}.pt".format(id, data_type))
        torch.save(sa, sa_path)
    else:
        X, y = data
    X_path = os.path.join(path, "{}_X_{}.pt".format(id, data_type))
    torch.save(X, X_path)
    y_path = os.path.join(path, "{}_y_{}.pt".format(id, data_type))
    torch.save(y, y_path)
