import torch
import pandas as pd
import time
import torchvision
from PIL import Image


def load_and_convert_images_to_tensors(subdata, data_key, sensitive_attributes, transforms):

    ### For each tuple (X, y):
    ### 1- Move sensitive attributes columns to the end of X
    ### 2- Load images and apply transform
    ### 3- Transform y to tensors
    ### 4- If needed, extract sensistive attributes columns and transform them to tensors

    X, y = subdata
    for sensitive_attribute in sensitive_attributes:
        sensitive_data = X.pop(sensitive_attribute)
        X = pd.concat([X, sensitive_data], axis=1)

    img_transform = torchvision.transforms.Compose(transforms)
    images_tensor = load_images_tensor_from_df(X, img_transform)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    if "train" in data_key:
        return images_tensor, y_tensor
    else:
        sa = X[sensitive_attributes]
        sa_tensor = torch.tensor(sa.values, dtype=torch.float32)
        return images_tensor, y_tensor, sa_tensor

def load_images_tensor_from_df(df, img_transform):
    print("LOADING images")
    start_time = time.time()

    images = []
    for img_path in df["image_path"]:
        img = Image.open(img_path)
        img_tensor = img_transform(img)
        images.append(img_tensor)
    images = torch.stack(images)

    end_time = time.time()
    tot_time = end_time - start_time
    print(tot_time)

    return images
