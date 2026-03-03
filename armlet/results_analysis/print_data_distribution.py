import random
import numpy as np
import pandas as pd

from armlet.data.datasets import load_Heart_dataset


SEED = 2


def print_data_distribution() -> None:

    dataset_name = "Heart"
    data_path = "./datasets/{}/raw_data/cardio_train.csv".format(dataset_name)
    sensitive_attributes = ['age', 'gender']

    np.random.seed(SEED)
    random.seed(SEED)

    data = load_Heart_dataset(
        path=data_path,
        sensitive_attributes=sensitive_attributes,
        train_size=0.8
    )

    all_data = []
    for key, val in data.items():
        x, y = val
        all_data.append(pd.concat([x, y], axis=1))
    df_all = pd.concat(all_data, axis=0)
    label = data["train"][1].columns[0]

    global_data_dist = []
    for sens_attr in sensitive_attributes:
        groupA_0 = df_all[(df_all[sens_attr] == 1) & (df_all[label] == 0)].shape[0] if not df_all[(df_all[sens_attr] == 0) & (df_all[label] == 0)].empty else 0
        global_data_dist.append([sens_attr, "groupA", 0, groupA_0])
        groupA_1 = df_all[(df_all[sens_attr] == 1) & (df_all[label] == 1)].shape[0] if not df_all[(df_all[sens_attr] == 0) & (df_all[label] == 1)].empty else 0
        global_data_dist.append([sens_attr, "groupA", 1, groupA_1])
        groupB_0 = df_all[(df_all[sens_attr] == 0) & (df_all[label] == 0)].shape[0] if not df_all[(df_all[sens_attr] == 1) & (df_all[label] == 0)].empty else 0
        global_data_dist.append([sens_attr, "groupB", 0, groupB_0])
        groupB_1 = df_all[(df_all[sens_attr] == 0) & (df_all[label] == 1)].shape[0] if not df_all[(df_all[sens_attr] == 1) & (df_all[label] == 1)].empty else 0
        global_data_dist.append([sens_attr, "groupB", 1, groupB_1])

    columns = ["sensitive_attribute", "group", "label", "n_samples"]
    global_data_dist = pd.DataFrame(global_data_dist, columns=columns)

    global_data_dist["ratio"] = (
        global_data_dist
        .groupby(["sensitive_attribute", "group"], group_keys=False)
        ["n_samples"]
        .apply(lambda df_group: df_group/df_group.sum())
    )
    print(global_data_dist)

    global_data_dist_wo_labels = (
        global_data_dist
        .groupby(["sensitive_attribute", "group"])
        .sum()
        .drop(columns=["label", "ratio"])
    )

    global_data_dist_wo_labels["ratio"] = (
        global_data_dist_wo_labels
        .groupby(["sensitive_attribute"], group_keys=False)
        ["n_samples"]
        .apply(lambda df_group: df_group/df_group.sum())
    )
    print(global_data_dist_wo_labels)

    print("ToT # of samples:", global_data_dist["n_samples"].sum() // global_data_dist["sensitive_attribute"].nunique())


if __name__ == "__main__":
    print_data_distribution()
