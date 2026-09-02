(config_federation_mode)=

# Federation configuration

In this page, we provide an overview of all the main configuration groups and values that can be used in **ARMLET** when running FL experiments (with the `federation` mode of **ARMLET**).
This way of managing configurations relies on YAML and is based on the one proposed by [Fluke](https://makgyver.github.io/fluke/configuration.html), which uses [Hydra](https://hydra.cc/).

```{eval-rst}

.. important::
  For the following or any other configuration explanations, we distinguish between two types of configuration elements:
  the **configuration values** (e.g., ``exp.seed``), which can be accessed using a ``.``,
  and the **configuration groups** (e.g., ``data/dataset``), which refer to a configuration file containing one or more config values and must be accessed using a ``/``.

```

In **ARMLET**, we compose both config groups and config values to run experiments with the desired configurations.
The config values are grouped into different categories (see below), which are organized into several subfolders within the ``ARMLET_DIR/armlet/configs`` folder.
These subfolders contain many pre-defined YAML config files that can be used when running experiments.

## Configuration categories

First, the general configuration groups used in each **ARMLET** mode (including the `federation` mode) are:

- [`armlet`](config_armlet): for choosing the mode of **ARMLET**;

- [`experiment`](config_experiment): for composing multiple configurations;

- [`hydra`](config_hydra): for managing Hydra;

- [`paths`](config_paths): for the general paths (datasets and outputs).

Then, the configuration groups specific to the `federation` mode are:

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
  Please look at the ``ARMLET_DIR/armlet/configs`` folder to explore the different config group possibilities.

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
  data_dir: ${paths.root_dir}/datasets
  output_dir: ${hydra:runtime.output_dir}

data:
  dataset:
    dataset_name: DC
    _target_: armlet.data.datasets.load_DC_dataset
    path: ./datasets/DC/raw_data/dutch_census_2001.txt
    sensitive_attributes: [age, gender]
    train_size: 0.8

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
        name: SGD
        lr: 0.001
        weight_decay: 0.01
      scheduler:
        name: StepLR
        gamma: 1
        step_size: 1

    server:
      weighted: true
      loss: ${method.hyperparameters.client.loss}

    model:
      _target_: armlet.utils.net.LogRegression
      input_size: null #auto
      num_classes: null #auto

eval:
  _target_: armlet.eval.evaluators.MultiCriteriaBinaryClassEval
  eval_every: 1
  pre_fit: true
  post_fit: true
  locals: true
  server: true
  sensitive_attributes: ${data.dataset.sensitive_attributes}
  metrics:
    fairness: armlet.eval.metrics.BinaryFairnessMetrics

logger:
  _target_: armlet.utils.log.ArmletLog
  comm_costs_tracker:
    track_every: ${protocol.n_rounds}
    clients_at_end_fit: true
    server_at_start_round: true

save: {}

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    Armlet<armlet>
    Experiment<experiment>
    Hydra<hydra>
    Paths<paths>
    Data<data>
    Eval<eval>
    Exp<exp>
    Logger<logger>
    Method<method>
    Protocol<protocol>
    Save<save>

```
