import torch
import pandas as pd
import time
import torchvision
from PIL import Image


def preprocess_image_data(
    subdata,
    data_key,
    transforms,
    dynamic_data_loading: bool=False,
    sensitive_attributes=[],
    y_dtype="float32",
    sa_dtype="float32",
    flatten_y: bool=False,
):

    X, y = subdata

    ### 1- Move sensitive attributes columns to the end of X
    for sensitive_attribute in sensitive_attributes:
        sensitive_data = X.pop(sensitive_attribute)
        X = pd.concat([X, sensitive_data], axis=1)

    ### 2- Load images and apply transform
    img_transform = torchvision.transforms.Compose(transforms)
    if dynamic_data_loading:
        images_tensor_list = load_images_tensor_from_df(X, img_transform)
    else:
        X["image_data"] = X["image_data"].apply(lambda x: img_transform(x))
        images_tensor_list = list(X["image_data"])
    images_tensor = torch.stack(images_tensor_list)

    ### 3- Transform y to tensors
    y_tensor = torch.tensor(y.values, dtype=getattr(torch, y_dtype))
    if flatten_y:
        y_tensor = y_tensor.flatten()

    ### 4- If needed, extract sensistive attributes columns and transform them to tensors
    if "train" in data_key:
        return images_tensor, y_tensor
    else:
        if sensitive_attributes:
            sa = X[sensitive_attributes]
            sa_tensor = torch.tensor(sa.values, dtype=getattr(torch, sa_dtype))
            return images_tensor, y_tensor, sa_tensor
        else:
            return images_tensor, y_tensor

def load_images_tensor_from_df(df, img_transform):
    print("LOADING images")
    start_time = time.time()

    images = []
    for img_path in df["image_path"]:
        img = Image.open(img_path)
        img_tensor = img_transform(img)
        images.append(img_tensor)

    end_time = time.time()
    tot_time = end_time - start_time
    print(tot_time)

    return images
