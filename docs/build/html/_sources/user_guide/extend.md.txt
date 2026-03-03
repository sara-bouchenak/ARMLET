# How to extend ``armlet``

## Add new datasets

1. Download and add the dataset in the folder `PROJECT_DIR/datasets/DATASET_NAME/`.

2. Create a Python file with a function that integrates the dataset loading along with its raw preprocessing.

3. Create a YAML config file with the name of the dataset in the folder `PROJECT_DIR/configs/data/dataset/` to provide the new option for the config group `data/dataset`.

## Add new ML models

1. Create a Python file with the new ML model.

2. The model can then be used in the experiments by replacing the config value `method.hyperparameters.model` (or for some model the loss function in the config value `method.hyperparameters.client.loss`).

## Add new metrics

[TODO]
