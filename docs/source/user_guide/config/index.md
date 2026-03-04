(config)=

# Configuration

In this page, we provide an overview of all the main configuration groups and values that can be used in **ARMLET** when running experiments.
This way of managing configurations is based on the one proposed by [Fluke](https://makgyver.github.io/fluke/configuration.html) and relies on YAML.
Nevertheless, we make several improvements to offer greater flexibility in managing experiments.

```{eval-rst}

.. important::
  For the following explanations, we distinguish between two types of configuration elements:
  the **configuration values** (e.g., ``exp.seed``), which can be accessed using a ``.``,
  and the **configuration groups** (e.g., ``data/dataset``), which refer to a configuration file containing one or more config values and must be accessed using a ``/``.

```

In **ARMLET**, we compose both config groups and config values to run experiments with the desired configurations.

## Main configuration categories

The main configuration groups are:

- [`data`](config_data): for everything related to data;

- [`eval`](config_eval): for the model evaluation;

- [`exp`](config_exp): generic settings for the experiment;

- [`hydra`](config_hydra): for managing Hydra;

- [`logger`](config_logger): logger configuration;

- [`method`](config_method): for the FL algorithm and its hyper-parameters;

- [`paths`](config_paths): for the general paths (data, log, output);

- [`protocol`](config_protocol): for the FL protocol;

- [`save`](config_save): saving configuration.

```{eval-rst}

.. seealso::
  All the essential config values are explained in the subpages, but the different options (i.e., config files) for each config group are not detailed.
  Please look at the ``ARMLET_DIR/configs`` folder to explore the different config group possibilities.

```

## Example

In the following, we provide an example of a YAML configuration file that can be directly used to run an experiment.
Note that these configuration values are detailed in the next documentation pages.

```yaml
# @package _global_

data:

  cleaning:
    missing_values:
      _target_: armlet.data.cleaning.missing_values.MissingValuesDataCleaningMethod
    name: default

  dataset:
    _target_: armlet.data.datasets.load_Adult_dataset
    dataset_name: Adult
    path: ./datasets/Adult/raw_data
    sensitive_attributes: ['age', 'gender', 'race']

  distribution:
    _target_: armlet.data.splitter.ArmletDataSplitter.iid

  others:
    client_split: 0.2
    client_val_split: 0.5
    keep_test: false
    sampling_perc: 1.0
    server_split: 0.0
    server_test: false
    server_test_union: true
    server_val_split: 0.0
    uniform_test: false

  seed: 42

eval:
  _target_: armlet.eval.evaluators.BinaryClassificationFairnessEval
  eval_every: 1
  locals: true
  post_fit: true
  pre_fit: true
  server: true

exp:
  device: cpu
  inmemory: true
  mode: federation
  seed: 42
  train: true

logger:
  _target_: armlet.utils.log.ArmletLog
  json_log_dir: ${paths.output_dir}

method:

  _target_: armlet.FL_pipeline.FL_algorithms.ArmletCentralizedFL

  hyperparameters:

    client:
      batch_size: 128
      local_epochs: 10
      loss:
        _target_: torch.nn.BCELoss
      optimizer:
        lr: 0.001
        name: SGD
        weight_decay: 0.01
      scheduler:
        gamma: 1
        name: StepLR
        step_size: 1

    model:
      _target_: armlet.utils.net.LogRegression
      input_size: 99
      num_classes: 1

    server:
      loss:
        _target_: torch.nn.BCELoss
      time_to_accuracy_target: null
      weighted: true

paths:
  data_dir: ./datasets
  log_dir: ./logs
  output_dir: ${hydra:runtime.output_dir}
  root_dir: .

protocol:
  eligible_perc: 1.0
  n_clients: 10
  n_rounds: 150

save: {}

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    Data<data>
    Eval<eval>
    Exp<exp>
    Hydra<hydra>
    Logger<logger>
    Method<method>
    Paths<paths>
    Protocol<protocol>
    Save<save>

```
