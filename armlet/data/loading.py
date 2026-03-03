"""
This module contains the functions for loading data dynamically (not from raw format).
"""

import pandas as pd
import os
import yaml
import torch


def load_data_from_folder(
    data_cfg_dict: dict,
    n_clients: int
):

    load_dir = data_cfg_dict["loading"]["load_dir"]
    assert os.path.exists(load_dir)
    del data_cfg_dict["loading"]
    if "saving" in data_cfg_dict.keys(): 
        del data_cfg_dict["saving"]

    with open(os.path.join(load_dir, "data_config.yaml"), 'r') as file:
        saved_data_cfg_dict = yaml.safe_load(file)
    if "loading" in data_cfg_dict.keys(): 
        del saved_data_cfg_dict["loading"]
    del saved_data_cfg_dict["saving"]

    if os.path.exists(os.path.join(load_dir, "data_cleaning_metrics.json")):
        is_data_cleaned = True
    else:
        is_data_cleaned = False
        del data_cfg_dict["cleaning"]
        del saved_data_cfg_dict["cleaning"]

    # Verify if the saved data have the same config
    data_cfg_dict["n_clients"] = n_clients
    assert data_cfg_dict == saved_data_cfg_dict

    data = {
        "clients_train": {},
        "clients_test": {},
        "clients_val": {},
        "server_test": None,
        "server_val": None,
    }

    is_tensor_data = "tensors" in load_dir.split(os.path.sep)[-1]

    for id_client in range(n_clients):
        id_client = "client_{}".format(id_client)
        client_path = os.path.join(load_dir, id_client)

        if is_tensor_data:
            data["clients_train"][id_client] = load_pt_data(client_path, id_client, "tr")
        else:
            data["clients_train"][id_client] = load_pkl_data(client_path, id_client, "tr")

        for (data_key, data_type, split_name) in [("clients_test", "te", "client_split"), ("clients_val", "val", "client_val_split")]:
            if data_cfg_dict["others"][split_name] > 0:
                if is_tensor_data:
                    data[data_key][id_client] = load_pt_data(client_path, id_client, data_type)
                else:
                    data[data_key][id_client] = load_pkl_data(client_path, id_client, data_type)
            else:
                data[data_key][id_client] = None

    if data_cfg_dict["others"]["server_test"] or data_cfg_dict["others"]["server_test_union"]:
        server_path = os.path.join(load_dir, "server")

        if is_tensor_data:
            data["server_test"] = load_pt_data(server_path, "server", "te")
        else:
            data["server_test"] = load_pkl_data(server_path, "server", "te")

        condition = data_cfg_dict["others"]["server_test_union"] and data_cfg_dict["others"]["client_val_split"] > 0
        if data_cfg_dict["others"]["server_val_split"] > 0 or condition:
            if is_tensor_data:
                data["server_val"] = load_pt_data(server_path, "server", "val")
            else:
                data["server_val"] = load_pkl_data(server_path, "server", "val")

    return data, is_data_cleaned

def load_pkl_data(path: str, id: str, data_type: str):
    X_path = os.path.join(path, "{}_X_{}.pkl".format(id, data_type))
    X = pd.read_pickle(X_path)
    y_path = os.path.join(path, "{}_y_{}.pkl".format(id, data_type))
    y = pd.read_pickle(y_path)
    return (X, y)

def load_pt_data(path: str, id: str, data_type: str):
    X_path = os.path.join(path, "{}_X_{}.pt".format(id, data_type))
    X = torch.load(X_path, weights_only=True)
    y_path = os.path.join(path, "{}_y_{}.pt".format(id, data_type))
    y = torch.load(y_path, weights_only=True)
    sa_path = os.path.join(path, "{}_sa_{}.pt".format(id, data_type))
    if os.path.exists(sa_path):
        sa = torch.load(sa_path, weights_only=True)
        data = (X, y, sa)
    else:
        data = (X, y)
    return data
