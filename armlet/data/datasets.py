"""
This module contains the functions for loading raw datasets.
"""

import numpy as np
import pandas as pd
import glob
import os
import pyreadstat
import pickle
import torch
import torchvision

from armlet.data.utils import dataframe_train_test_split


def load_ARS_dataset(
    path: str,
    sensitive_attributes: list[str],
    train_size: float,
) -> dict:

    ### data source: https://archive.ics.uci.edu/dataset/427/activity+recognition+with+healthy+older+people+using+a+batteryless+wearable+sensor

    list_df = []
    files_path = glob.glob(os.path.join(path, "*", "d*"))
    names = ["time", "x-axis", "y-axis", "z-axis", "sensor", "rssi", "phase", "frequency", "activity"]
    for file_path in files_path:
        df_file = pd.read_csv(file_path, header=None, names=names)
        df_file["gender"] = file_path[-1]
        df_file["room"] = int(file_path.split(os.path.sep)[4][1])
        list_df.append(df_file)
    df = pd.concat(list_df, axis=0, ignore_index=True)

    label = "activity"
    df[label] = df[label].apply(lambda x: 1 if x == 3 else 0)
    assert df[label].nunique() == 2

    for sensitive_attribute in sensitive_attributes:
        assert sensitive_attribute in df.columns
        if sensitive_attribute == "gender":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == "M" else False)
        assert df[sensitive_attribute].nunique() == 2

    bool_col = [label]
    df[bool_col] = df[bool_col].astype('boolean')

    categ_col_for_ohe = ["room", "sensor"]
    df[categ_col_for_ohe] = df[categ_col_for_ohe].astype('object')

    X = df.drop([label, "time"], axis=1)
    y = df[[label]]

    X_train, X_test, y_train, y_test = dataframe_train_test_split(
        X, y, train_size=train_size
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}

def load_Adult_dataset(
    path: str,
    sensitive_attributes: list[str],
) -> dict:

    ### data source: https://archive.ics.uci.edu/dataset/2/adult

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'income']

    train_data_path = os.path.join(path, "adult.data")
    test_data_path = os.path.join(path, "adult.test")

    data_dict = {}
    for set_name, data_path in [("train", train_data_path), ("test", test_data_path)]:

        if set_name == "train":
            df = pd.read_csv(data_path, names=columns, sep=', ', engine="python")
        else:
            df = pd.read_csv(data_path, names=columns, sep=', ', engine="python", header=0)

        df = df.replace("?", np.nan)

        label = "income"
        df[label] = df[label].apply(lambda x: 1 if (x == ">50K" or x == ">50K.") else 0)
        assert df[label].nunique() == 2

        df = df.rename(columns={"sex": "gender"})
        for sensitive_attribute in sensitive_attributes:
            assert sensitive_attribute in df.columns
            if sensitive_attribute == "race":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if (x == "White" or x == "Asian-Pac-Islander") else False)
            elif sensitive_attribute == "gender":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == "Male" else False)
            elif sensitive_attribute == "age":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if (x >= 30 and x <= 60) else False)
            assert df[sensitive_attribute].nunique() == 2

        bool_col = [label]
        df[bool_col] = df[bool_col].astype('boolean')
        
        num_categ_col = ["education-num"]
        df[num_categ_col] = df[num_categ_col].astype('category')

        X = df.drop(label, axis=1)
        y = df[[label]]
        data_dict[set_name] = (X, y)

    return data_dict

def load_Heart_dataset(
    path: str,
    sensitive_attributes: list[str],
    train_size: float,
):

    ### data source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

    df = pd.read_csv(path, sep=";")

    label = "cardio"
    assert df[label].nunique() == 2

    for sensitive_attribute in sensitive_attributes:
        assert sensitive_attribute in df.columns
        if sensitive_attribute == "gender":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == 2 else False)
        elif sensitive_attribute == "age":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x > 45*365 else False)
        assert df[sensitive_attribute].nunique() == 2

    bool_col = [label, "smoke", "alco", "active"]
    df[bool_col] = df[bool_col].astype('boolean')

    int_categ_col = ["cholesterol", "gluc"]
    df[int_categ_col] = df[int_categ_col].astype('category')

    X = df.drop([label, "id"], axis=1)
    y = df[[label]]

    X_train, X_test, y_train, y_test = dataframe_train_test_split(
        X, y, train_size=train_size
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}

def load_KDD_dataset(
    path: str,
    sensitive_attributes: list[str],
):

    ### data source: https://archive.ics.uci.edu/dataset/117/census+income+kdd

    columns = ['AAGE', 'ACLSWKR', 'ADTINK', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL',
       'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEM',
       'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'GAPLOSS', 'DIVVAL', 'FILESTAT',
       'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MARSUPWRT', 'MIGMTR1',
       'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT',
       'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA',
       'VETYN', 'WKSWORK', 'year', 'income']

    train_data_path = os.path.join(path, "census-income.data")
    test_data_path = os.path.join(path, "census-income.test")

    data_dict = {}
    for set_name, data_path in [("train", train_data_path), ("test", test_data_path)]:

        df = pd.read_csv(data_path, names=columns, sep=', ', engine="python", na_filter=False)

        df = df.replace("?", np.nan)

        label = "income"
        df[label] = df[label].apply(lambda x: 1 if x == "50000+." else 0)
        assert df[label].nunique() == 2

        df = df.rename(columns={"ARACE": "race", "ASEX": "gender", "AAGE": "age"})
        for sensitive_attribute in sensitive_attributes:
            assert sensitive_attribute in df.columns
            if sensitive_attribute == "race":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if (x == "White" or x == "Asian or Pacific Islander") else False)
            elif sensitive_attribute == "gender":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == "Male" else False)
            elif sensitive_attribute == "age":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if (x >= 30 and x <= 60) else False)
            assert df[sensitive_attribute].nunique() == 2

        bool_col = [label]
        df[bool_col] = df[bool_col].astype('boolean')

        int_categ_col = ["ADTINK", "ADTOCC", "NOEMP"]
        df[int_categ_col] = df[int_categ_col].astype('category')

        categ_col_for_ohe = ["SEOTR", "VETYN"]
        df[categ_col_for_ohe] = df[categ_col_for_ohe].astype('object')

        X = df.drop(label, axis=1)
        y = df[[label]]
        data_dict[set_name] = (X, y)

    return data_dict

def load_MEPS_dataset(
    path: str,
    sensitive_attributes: list[str],
    train_size: float,
):

    ### data source: "Data File, SAS transport format --> ZIP" at https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181

    df, _ = pyreadstat.read_xport(path)

    label = "UTILIZATION"
    label_func = lambda x: 1 if (x['OBTOTV15'] + x['OPTOTV15'] + x['ERTOT15'] + x['IPNGTD15'] + x['HHTOTD15']) >= 10 else 0
    df[label] = df.apply(label_func, axis=1)
    assert df[label].nunique() == 2

    df = df.rename(columns={"RACEV2X": "race", "SEX": "gender"})
    for sensitive_attribute in sensitive_attributes:
        assert sensitive_attribute in df.columns
        if sensitive_attribute == "race":
            sa_func = lambda x: True if (x['HISPANX'] == 2 and x['race'] == 1) else False
            df[sensitive_attribute] = df.apply(sa_func, axis=1)
        elif sensitive_attribute == "gender":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == 1 else False)
        assert df[sensitive_attribute].nunique() == 2

    columns_mask_1 = ['REGION53', 'AGE53X', 'MARRY53X', 'ASTHDX']
    mask_1 = (df[columns_mask_1] >= 0).all(axis=1)
    df = df[mask_1]

    columns_mask_2 = ['FTSTU53X', 'ACTDTY53', 'HONRDC53', 'RTHLTH53', 'MNHLTH53', 'HIBPDX', 'CHDDX', 'ANGIDX',
                    'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON53', 'CHOLDX', 'CANCERDX', 'DIABDX', 'HIDEG',
                    'JTPAIN53', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT53', 'WLKLIM53', 'EDUCYR',
                    'ACTLIM53', 'SOCLIM53', 'COGLIM53', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PHQ242', 'EMPST53',
                    'POVCAT15', 'INSCOV15']
    mask_2 = (df[columns_mask_2] >= -1).all(axis=1)
    df = df[mask_2]

    df = df.reset_index(drop=True)

    columns_to_keep = ['REGION53', 'AGE53X', 'gender', 'race', 'MARRY53X', 'FTSTU53X', 'ACTDTY53', 'HONRDC53',
                        'RTHLTH53', 'MNHLTH53', 'HIBPDX', 'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX',
                        'EMPHDX', 'CHBRON53', 'CHOLDX','CANCERDX','DIABDX', 'JTPAIN53', 'ARTHDX', 'ARTHTYPE',
                        'ASTHDX', 'ADHDADDX', 'PREGNT53', 'WLKLIM53', 'ACTLIM53', 'SOCLIM53', 'COGLIM53',
                        'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST53',
                        'POVCAT15', 'INSCOV15', 'UTILIZATION', 'PERWT15F']
    df = df[columns_to_keep]

    bool_col = [label]
    df[bool_col] = df[bool_col].astype('boolean')

    categ_col_for_ohe = ['REGION53', 'MARRY53X', 'FTSTU53X', 'ACTDTY53', 'HONRDC53', 'RTHLTH53', 'MNHLTH53',
                    'HIBPDX', 'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON53', 'CHOLDX',
                    'CANCERDX', 'DIABDX', 'JTPAIN53', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT53',
                    'WLKLIM53', 'ACTLIM53', 'SOCLIM53', 'COGLIM53', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                    'PHQ242', 'EMPST53', 'POVCAT15', 'INSCOV15']
    df[categ_col_for_ohe] = df[categ_col_for_ohe].astype('int32').astype('object')

    X = df.drop([label], axis=1)
    y = df[[label]]

    X_train, X_test, y_train, y_test = dataframe_train_test_split(
        X, y, train_size=train_size
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}

def load_DC_dataset(
    path: str,
    sensitive_attributes: list[str],
    train_size: float,
):

    ### data source: https://github.com/tailequy/fairness_dataset --> "Dutch_census" folder

    df = pd.read_csv(path)

    label = "occupation"
    df[label] = df[label].apply(lambda x: 1 if x == "2_1" else 0)
    assert df[label].nunique() == 2

    df = df.rename(columns={"sex": "gender"})
    for sensitive_attribute in sensitive_attributes:
        assert sensitive_attribute in df.columns
        if sensitive_attribute == "age":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x < 11 else False)
        elif sensitive_attribute == "gender":
            df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == 1 else False)
        assert df[sensitive_attribute].nunique() == 2

    household_size_mapping = {111: 1, 112: 2, 113: 3, 114: 4, 125: 5, 126: 6}
    df["household_size"] = df["household_size"].apply(lambda x: household_size_mapping[x])

    bool_col = [label]
    df[bool_col] = df[bool_col].astype('boolean')

    int_categ_col = ["household_size", "edu_level"]
    df[int_categ_col] = df[int_categ_col].astype('category')

    categ_col_for_ohe = ["household_position", "prev_residence_place", "citizenship", "country_birth",
                        "economic_status", "cur_eco_activity", "marital_status"]
    df[categ_col_for_ohe] = df[categ_col_for_ohe].astype('object')

    X = df.drop([label], axis=1)
    y = df[[label]]

    X_train, X_test, y_train, y_test = dataframe_train_test_split(
        X, y, train_size=train_size
    )

    return {"train": (X_train, y_train), "test": (X_test, y_test)}

def load_CelebA_dataset(
    path: str,
    sensitive_attributes: list[str],
    **kwargs,
):

    ### data source: https://docs.pytorch.org/vision/0.17/generated/torchvision.datasets.CelebA.html

    attr_file_path = os.path.join(path, "list_attr_celeba.txt")
    df_attr = pd.read_csv(attr_file_path, sep=r"\s+", header=1)
    df_attr = df_attr.apply(lambda val: (val + 1) // 2)

    eval_partition_path = os.path.join(path, "list_eval_partition.txt")
    df_eval_partition = pd.read_csv(eval_partition_path, sep=r"\s+", index_col=0, names=["eval_partition"])

    df = pd.concat([df_attr, df_eval_partition], axis=1)
    df = df.reset_index(names="image_name")

    label = "Smiling"
    assert df[label].nunique() == 2

    df = df.rename(columns={"Young": "age", "Male": "gender"})
    for sensitive_attribute in sensitive_attributes:
        assert sensitive_attribute in df.columns
        assert df[sensitive_attribute].nunique() == 2

    bool_col = [label] + sensitive_attributes
    for bool_c in bool_col:
        df[bool_c] = df[bool_c].astype('boolean')

    img_dir =  os.path.join(path, "img_align_celeba")
    df["image_path"] = df["image_name"].apply(lambda val: os.path.join(img_dir, val))

    mask_train = df["eval_partition"] == 0
    X_train = df.loc[mask_train][["image_path"]+sensitive_attributes]
    y_train = df.loc[mask_train][[label]]
    X_test = df.loc[~mask_train][["image_path"]+sensitive_attributes]
    y_test = df.loc[~mask_train][[label]]

    return {"train": (X_train, y_train), "test": (X_test, y_test)}

def load_FairFace_dataset(
    path: str,
    sensitive_attributes: list[str],
    **kwargs,
):
    
    ### data source: https://github.com/dchen236/FairFace

    img_dir =  os.path.join(path, "fairface-img-margin025-trainval")
    train_data_path = os.path.join(path, "fairface_label_train.csv")
    test_data_path = os.path.join(path, "fairface_label_val.csv")

    data_dict = {}
    for set_name, data_path in [("train", train_data_path), ("test", test_data_path)]:

        df = pd.read_csv(data_path, header=0)

        label = "gender"
        df[label] = df[label].apply(lambda x: 1 if x == "Female" else 0)
        assert df[label].nunique() == 2

        for sensitive_attribute in sensitive_attributes:
            assert sensitive_attribute in df.columns
            if sensitive_attribute == "age":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x in ["40-49", "50-59", "60-69", "0-2"] else False)
            elif sensitive_attribute == "race":
                df[sensitive_attribute] = df[sensitive_attribute].apply(lambda x: True if x == "Middle Eastern" else False)
            assert df[sensitive_attribute].nunique() == 2

        bool_col = [label]
        df[bool_col] = df[bool_col].astype('boolean')

        df["image_path"] = df["file"].apply(lambda file: os.path.join(img_dir, file))

        X = df[["image_path"]+sensitive_attributes]
        y = df[[label]]
        data_dict[set_name] = (X, y)

    return data_dict


def unpickle(file_path):
    with open(file_path, "rb") as fo:
        data_dict = pickle.load(fo, encoding="latin1")
    return data_dict


def load_CIFAR_10_dataset(path: str, **kwargs):
    file_names = [f"data_batch_{i}" for i in range(1, 6)]
    train_data_paths = [os.path.join(path, file_name) for file_name in file_names]

    train_data_dicts = [unpickle(train_path) for train_path in train_data_paths]
    x_train = np.vstack([batch["data"] for batch in train_data_dicts])
    x_train = pd.DataFrame({"image_data": list(x_train)})
    x_train["image_data"] = x_train["image_data"].apply(lambda x: torch.from_numpy(x))
    y_train = []
    for batch in train_data_dicts:
        y_train.extend(batch["labels"])
    y_train = pd.DataFrame({"label": y_train}, dtype="int64")

    test_path = os.path.join(path, "test_batch")
    test_data_dict = unpickle(test_path)
    x_test = pd.DataFrame({"image_data": list(test_data_dict["data"])})
    x_test["image_data"] = x_test["image_data"].apply(lambda x: torch.from_numpy(x))
    y_test = pd.DataFrame({"label": test_data_dict["labels"]}, dtype="int64")

    return {"train": (x_train, y_train), "test": (x_test, y_test)}

def load_Purchase_dataset(
    path: str,
    train_data_file: str = "purchase_train_data.npy",
    train_labels_file: str = "purchase_train_labels.npy",
    test_data_file: str = "purchase_test_data.npy",
    test_labels_file: str = "purchase_test_labels.npy",
    **kwargs,
) -> dict:

    train_X = np.load(os.path.join(path, train_data_file))
    train_y = np.load(os.path.join(path, train_labels_file)).astype(np.int64)
    test_X = np.load(os.path.join(path, test_data_file))
    test_y = np.load(os.path.join(path, test_labels_file)).astype(np.int64)

    feature_names = [f"feature_{idx}" for idx in range(train_X.shape[1])]
    return {
        "train": (
            pd.DataFrame(train_X, columns=feature_names),
            pd.DataFrame(train_y, columns=["label"]),
        ),
        "test": (
            pd.DataFrame(test_X, columns=feature_names),
            pd.DataFrame(test_y, columns=["label"]),
        ),
    }

def load_EuroSAT_dataset(
    path: str,
    train_size: float = 0.8,
    download: bool = True,
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
    split_strategy: str = "torch_random_split",
    **kwargs,
) -> dict:

    try:
        dataset = torchvision.datasets.EuroSAT(path, download=download)
    except Exception as exc:
        if not download:
            raise
        import ssl

        print(f"[EuroSAT] download failed. Retrying without SSL verification. Error: {exc}")
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = torchvision.datasets.EuroSAT(path, download=True)

    samples = getattr(dataset, "samples", None)
    if samples is None:
        samples = [(path_, target) for path_, target in zip(dataset._image_files, dataset.targets)]

    X = pd.DataFrame({"image_path": [sample_path for sample_path, _ in samples]})
    y = pd.DataFrame({"label": [int(label) for _, label in samples]})

    train_idx, test_idx = _train_test_indices(
        labels=y["label"].to_numpy(),
        train_size=float(train_size),
        strategy=split_strategy,
    )

    if train_subset_size is not None and train_subset_size > 0 and train_subset_size < len(train_idx):
        train_idx = np.random.choice(train_idx, size=int(train_subset_size), replace=False)
    if test_subset_size is not None and test_subset_size > 0 and test_subset_size < len(test_idx):
        test_idx = np.random.choice(test_idx, size=int(test_subset_size), replace=False)

    return {
        "train": (
            X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
        ),
        "test": (
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True),
        ),
    }

def _train_test_indices(labels: np.ndarray, train_size: float, strategy: str):
    if strategy == "torch_random_split":
        n_train = int(train_size * len(labels))
        indices = torch.randperm(len(labels)).numpy()
        return indices[:n_train], indices[n_train:]
    if strategy == "stratified":
        return _stratified_train_test_indices(labels, train_size=train_size)
    raise ValueError(f"Unknown EuroSAT split_strategy={strategy!r}.")

def _stratified_train_test_indices(labels: np.ndarray, train_size: float):
    train_indices = []
    test_indices = []
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        np.random.shuffle(class_indices)
        n_train = int(train_size * len(class_indices))
        train_indices.extend(class_indices[:n_train].tolist())
        test_indices.extend(class_indices[n_train:].tolist())

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
    
def load_Speech_commands_dataset(
    path: str,
    max_samples_per_class: int | None = None,
    max_total_samples: int | None = None,
    **kwargs,
):

    ### data source:http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    ### extract the tar.gz file into "speech_commands" in the datasets dir, e.g. armlet/data/datasets/speech_commands

    assert os.path.isdir(path), "Speech Commands dataset directory does not exist."

    validation_file = os.path.join(path, "validation_list.txt")
    testing_file = os.path.join(path, "testing_list.txt")

    with open(validation_file, "r") as f:
        validation_paths = set(line.strip() for line in f if line.strip())

    with open(testing_file, "r") as f:
        testing_paths = set(line.strip() for line in f if line.strip()) 

    classes = [
        "backward","bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
        "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
        "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", 
        "two", "up", "visual", "wow", "yes", "zero",
    ]
    label_mapping = {
        class_name: class_id
        for class_id, class_name in enumerate(classes)
    }

    data_list = []
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        files_path = sorted(glob.glob(os.path.join(class_path, "*.wav")))
        for audio_path in files_path:
            relative_path = os.path.relpath(
                audio_path,
                path,
            ).replace(os.path.sep, "/")

            if relative_path in testing_paths:
                set_name = "test"
            elif relative_path in validation_paths:
                set_name = "validation"
            else:
                set_name = "train"

            data_list.append(
                {
                    "audio_path": audio_path,
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "label": label_mapping[class_name],
                    "set": set_name,
                }
            )

    df = pd.DataFrame(data_list)

    if max_samples_per_class is not None:
        df = (
            df.groupby("class_name", group_keys=False)
            .head(max_samples_per_class)
            .reset_index(drop=True)
        )

    if max_total_samples is not None:
        df = df.head(max_total_samples).reset_index(drop=True)

    mask_train = df["set"].isin(["train", "validation"])
    mask_test = df["set"] == "test"

    X_train = df.loc[mask_train, ["audio_path"]].reset_index(drop=True)
    y_train = df.loc[mask_train, ["label"]].astype("int64").reset_index(drop=True)

    X_test = df.loc[mask_test, ["audio_path"]].reset_index(drop=True)
    y_test = df.loc[mask_test, ["label"]].astype("int64").reset_index(drop=True)

    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test)
    } 

def load_AudioMNIST_dataset(
    path: str,
    test_speakers: list[str | int] | None = None,
    max_samples_per_class: int | None = None,
    max_total_samples: int | None = None,
    **kwargs,
):
    ### data source: https://github.com/soerenab/AudioMNIST/tree/master/data
    ### curl the data dir from the repo above into AudioMNIST in the datasets dir, e.g. armlet/data/datasets/AudioMNIST/data
    
    assert os.path.isdir(path), "AudioMNIST dataset directory does not exist."

    if test_speakers is None:
        test_speakers = ["56", "57", "58", "59", "60"]
    test_speakers = {str(speaker_id).zfill(2) for speaker_id in test_speakers}

    files_path = sorted(glob.glob(os.path.join(path, "*", "*.wav")))
    assert files_path, f"No AudioMNIST WAV files found under {path}."

    data_list = []
    for audio_path in files_path:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        digit, speaker_id, recording_id = filename.split("_")
        speaker_id = speaker_id.zfill(2)

        data_list.append(
            {
                "audio_path": audio_path,
                "speaker_id": speaker_id,
                "recording_id": int(recording_id),
                "label": int(digit),
                "set": "test" if speaker_id in test_speakers else "train",
            }
        )

    df = pd.DataFrame(data_list)

    if max_samples_per_class is not None:
        df = (
            df.groupby(["set", "label"], group_keys=False)
            .head(max_samples_per_class)
            .reset_index(drop=True)
        )

    if max_total_samples is not None:
        df = df.head(max_total_samples).reset_index(drop=True)

    assert df["label"].min() >= 0
    assert df["label"].max() < 10

    mask_train = df["set"] == "train"
    mask_test = df["set"] == "test"
    assert mask_train.any(), "AudioMNIST train split is empty."
    assert mask_test.any(), "AudioMNIST test split is empty."

    X_train = df.loc[
        mask_train, ["audio_path", "speaker_id", "recording_id"]
    ].reset_index(drop=True)
    y_train = df.loc[mask_train, ["label"]].astype("int64").reset_index(drop=True)

    X_test = df.loc[
        mask_test, ["audio_path", "speaker_id", "recording_id"]
    ].reset_index(drop=True)
    y_test = df.loc[mask_test, ["label"]].astype("int64").reset_index(drop=True)

    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }
