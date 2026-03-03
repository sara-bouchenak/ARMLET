"""
This module contains the functions for loading raw datasets.
"""

import numpy as np
import pandas as pd
import glob
import os
import pyreadstat

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
