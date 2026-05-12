(config_federation_mode)=

# Configuration

In this page, we provide an overview of all the main configuration groups and values that can be used in **ARMLET** when running FL experiments (with the `federation` mode of `armlet`).

See [Configuration prerequisites](config) for any information about the general configuration groups and values of **ARMLET**.

## Main configuration categories of the `federation` mode

The main configuration groups related to the `federation` mode are:

- [`data`](config_data): for everything related to data;

- [`eval`](config_eval): for the model evaluation;

- [`exp`](config_exp): generic settings for the FL experiment;

- [`logger`](config_logger): logger configuration;

- [`method`](config_method): for the FL algorithm and its hyper-parameters;

- [`protocol`](config_protocol): for the FL protocol;

- [`save`](config_save): for saving models.

```{eval-rst}

.. seealso::
  All the essential config values are explained in the subpages, but the different options (i.e., config files) for each config group are not detailed.
  Please look at the ``ARMLET_DIR/configs`` folder to explore the different config group possibilities.

```

## Example

In the following, we provide an example of a YAML configuration file that can be directly used to run an FL experiment with the `federation` mode.
Note that these configuration values are detailed in the next documentation pages.

```yaml
# @package _global_

armlet:
  mode: federation

paths:
  root_dir: .
  data_dir: ./datasets
  log_dir: ./logs
  output_dir: ${hydra:runtime.output_dir}


data:

  dataset:
    dataset_name: Adult
    _target_: armlet.data.datasets.load_Adult_dataset
    path: ./datasets/Adult/raw_data
    sensitive_attributes: [age, gender, race]

  splitter:
    distribution:
      _target_: armlet.data.splitter.ArmletDataSplitter.iid
    _target_: armlet.data.splitter.ArmletDataSplitter
    client_split: 0.2
    client_val_split: 0.5
    keep_test: false
    server_test: false
    server_test_union: true
    server_split: 0.0
    server_val_split: 0.0
    uniform_test: false

  cleaning:
    missing_values:
      _target_: armlet.data.cleaning.missing_values.RemoveMV
    name: default

  processing:
    one_hot_encoding:
      _target_: armlet.data.processing.feature_encoding.one_hot_encoding_pipeline
      _apply_directly_to_subdata_: false
    conversion_to_num:
      _target_: armlet.data.processing.format_conversion.convert_bool_and_cat_to_num
      _apply_directly_to_subdata_: true
    normalization:
      _target_: armlet.data.processing.normalization.normalization_pipeline
      _apply_directly_to_subdata_: false
      cols_to_exclude: ${data.dataset.sensitive_attributes}
    conversion_to_tensors:
      _target_: armlet.data.processing.format_conversion.convert_dataframes_to_tensors
      _apply_directly_to_subdata_: true
      sensitive_attributes: ${data.dataset.sensitive_attributes}

  others:
    sampling_perc: 1.0

  seed: 42

exp:
  device: cpu
  seed: 42
  inmemory: true
  train: true

protocol:
  eligible_perc: 1.0
  n_clients: 10
  n_rounds: 150

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

eval:
  _target_: armlet.eval.evaluators.MultiCriteriaBinaryClassEval
  eval_every: 1
  locals: true
  metrics:
    fairness: armlet.eval.metrics.BinaryFairnessMetrics
  post_fit: true
  pre_fit: true
  server: true

logger:
  _target_: armlet.utils.log.ArmletLog
  json_log_dir: ${paths.output_dir}

save: {}

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    Data<data>
    Eval<eval>
    Exp<exp>
    Logger<logger>
    Method<method>
    Protocol<protocol>
    Save<save>

```
