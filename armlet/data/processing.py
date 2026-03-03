"""
This module contains the data processing functions.
"""

import torch
import pandas as pd
import time
import torch
import torchvision
from PIL import Image

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from fluke.data import FastDataLoader


def one_hot_encoding(data):

    list_df = []
    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    list_df.append(X)
        else:
            if subdata != None:
                X, y = subdata
                list_df.append(X)
    df = pd.concat(list_df, axis=0, ignore_index=True)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "image_path"] # no ohe for "image_path" column
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(df[categorical_cols])

    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    one_hot_array = one_hot_encoder.transform(X[categorical_cols]).toarray()
                    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_encoder.get_feature_names_out())
                    X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
                    data[key][id_client] = (X_one_hot, y)
        else:
            if subdata != None:
                X, y = subdata
                one_hot_array = one_hot_encoder.transform(X[categorical_cols]).toarray()
                one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_encoder.get_feature_names_out())
                X_one_hot = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), one_hot_df], axis=1)
                data[key] = (X_one_hot, y)

    return data

def convert_bool_and_cat_to_num(data):
    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    cols = X.select_dtypes(include=["boolean", "category"]).columns.tolist()
                    X[cols] = X[cols].astype('int8')
                    y = y.astype('int8')
                    data[key][id_client] = (X, y)
        else:
            if subdata != None:
                X, y = subdata
                cols = X.select_dtypes(include=["boolean", "category"]).columns.tolist()
                X[cols] = X[cols].astype('int8')
                y = y.astype('int8')
                data[key] = (X, y)
    return data

def normalization(data, sensitive_attributes):

    list_df = [X for (X, y) in data["clients_train"].values()]
    df = pd.concat(list_df, axis=0, ignore_index=True)

    norm_cols = df.columns.tolist()
    norm_cols = [col for col in norm_cols if col not in sensitive_attributes]
    norm_cols = [col for col in norm_cols if col != "image_path"] # no normalization for "image_path" column

    if norm_cols != []:

        scaler = StandardScaler()
        scaler.fit(df[norm_cols])

        for key, subdata in data.items():
            if "clients" in key:
                for id_client, client_data in subdata.items():
                    if client_data != None:
                        X, y = client_data
                        X[norm_cols] = scaler.transform(X[norm_cols])
                        data[key][id_client] = (X, y)
            else:
                if subdata != None:
                    X, y = subdata
                    X[norm_cols] = scaler.transform(X[norm_cols])
                    data[key] = (X, y)

    return data

def convert_dataframes_to_tensors(data, sensitive_attributes):

    ### For each tuple (X, y):
    ### 1- Move sensitive attributes columns to the end of X
    ### 2- Transform X and y to tensors

    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    for sensitive_attribute in sensitive_attributes:
                        sensitive_data = X.pop(sensitive_attribute)
                        X = pd.concat([X, sensitive_data], axis=1)
                    X_tensor = torch.tensor(X.values, dtype=torch.float32)
                    y_tensor = torch.tensor(y.values, dtype=torch.float32)
                    data[key][id_client] = (X_tensor, y_tensor)
        else:
            if subdata != None:
                X, y = subdata
                for sensitive_attribute in sensitive_attributes:
                    sensitive_data = X.pop(sensitive_attribute)
                    X = pd.concat([X, sensitive_data], axis=1)
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                y_tensor = torch.tensor(y.values, dtype=torch.float32)
                data[key] = (X_tensor, y_tensor)
    return data

def load_and_convert_images_to_tensors(data, cfg_dataset):

    ### For each tuple (X, y):
    ### 1- Move sensitive attributes columns to the end of X
    ### 2- Extract sensistive attributes columns and transform them to tensors
    ### 3- Load images and apply transform
    ### 4- Transform y to tensors

    transform_list = []
    if "new_image_size" in cfg_dataset.keys():
        new_image_size = tuple(cfg_dataset.new_image_size)
        transform_list.append(torchvision.transforms.Resize(new_image_size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform_list.append(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = torchvision.transforms.Compose(transform_list)

    for key, subdata in data.items():
        if "clients" in key:
            for id_client, client_data in subdata.items():
                if client_data != None:
                    X, y = client_data
                    for sensitive_attribute in cfg_dataset.sensitive_attributes:
                        sensitive_data = X.pop(sensitive_attribute)
                        X = pd.concat([X, sensitive_data], axis=1)
                    sa = X[cfg_dataset.sensitive_attributes]
                    sa_tensor = torch.tensor(sa.values, dtype=torch.float32)
                    images_tensor = load_images_tensor_from_df(X, transform)
                    y_tensor = torch.tensor(y.values, dtype=torch.float32)
                    if "train" in key:
                        data[key][id_client] = (images_tensor, y_tensor)
                    else:
                        data[key][id_client] = (images_tensor, y_tensor, sa_tensor)
        else:
            if subdata != None:
                X, y = subdata
                for sensitive_attribute in cfg_dataset.sensitive_attributes:
                    sensitive_data = X.pop(sensitive_attribute)
                    X = pd.concat([X, sensitive_data], axis=1)
                sa = X[cfg_dataset.sensitive_attributes]
                sa_tensor = torch.tensor(sa.values, dtype=torch.float32)
                images_tensor = load_images_tensor_from_df(X, transform)
                y_tensor = torch.tensor(y.values, dtype=torch.float32)
                data[key] = (images_tensor, y_tensor, sa_tensor)

    return data

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

def load_images_tensor_from_df(df, transform):
    print("LOADING images")
    start_time = time.time()

    images = []
    for img_path in df["image_path"]:
        img = Image.open(img_path)
        img_tensor = transform(img)
        images.append(img_tensor)
    images = torch.stack(images)

    end_time = time.time()
    tot_time = end_time - start_time
    print(tot_time)

    return images
