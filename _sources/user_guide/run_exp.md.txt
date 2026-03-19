(run_exp)=

# Running experiments

**ARMLET** is based on [Hydra](https://hydra.cc/) for simplifying the launch of experiments and ensuring high reproductibily with config files.
In the following, we provide examples that illustrate how to efficiently run experiments with our framework, via simple command-line (A) or through experiment config files (B).

```{eval-rst}

.. seealso::
	Please look at `Hydra documentation <https://hydra.cc/docs/intro/>`_ for further comprehension about Hydra basics.

```

## A- Run experiment(s) from command-line

Run `armlet` to launch a single experiment with the default configuration values.

```bash
armlet
```

Config values (e.g., `exp.seed`) or config groups (e.g., `data/dataset`) can be directly overrided from the command line.
Note that `+` need to be added before the config values or groups that do not have default values.

```bash
armlet exp.seed=1 data/dataset=ars +data/loading=default
```

```{eval-rst}

.. seealso::
	For more information about configuration files, config values, or config groups, see :ref:`Configuration <config>`.

```

Multiple experiments can also be run sequentially by first adding the option `-m` to the command line and then providing multiple values (separated by a comma) for the config values or the config groups.
In this case, all combinations of values to be sweeped will be crossed to launch a batch of experiments.

```bash
armlet -m exp.seed=1,2,3,4,5 data/dataset=ars,heart
```

## B- Run experiment(s) from a YAML experiment config file

1. Create a YAML configuration file in `PROJECT_DIR/configs/experiment/` by copying, pasting, and overriding `ARMLET_DIR/configs/experiment/template.yaml`.

```{eval-rst}

.. tip::
  Config files saved during a past experiment (``EXP_PATH/config.yaml``) can notably be copied and pasted to ``PROJECT_DIR/configs/experiment/`` to reproduce the experiment.
  In that case, add ``# @package _global_`` at the beggining of the file just as in the following example.

```

**YAML configuration file example:**

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

  processing:
    one_hot_encoding:
      _apply_directly_to_subdata_: false
      _target_: armlet.data.processing.feature_encoding.one_hot_encoding_pipeline
    conversion_to_num:
      _apply_directly_to_subdata_: true
      _target_: armlet.data.processing.format_conversion.convert_bool_and_cat_to_num
    normalization:
      _apply_directly_to_subdata_: false
      _target_: armlet.data.processing.normalization.normalization_pipeline
      cols_to_exclude: ${data.dataset.sensitive_attributes}
    conversion_to_tensors:
      _apply_directly_to_subdata_: true
      _target_: armlet.data.processing.format_conversion.convert_dataframes_to_tensors
      sensitive_attributes: ${data.dataset.sensitive_attributes}

  seed: 42

eval:
  _target_: armlet.eval.evaluators.MultiCriteriaBinaryClassEval
  eval_every: 1
  locals: true
  metrics:
    fairness: armlet.eval.metrics.BinaryFairnessMetrics
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

.. important::
  Just as command line mode, config values that are specified in the experiment config file override the default configuration.
  Thus, config values that are not provided in this file will be set to their default values.

```

For greater efficiency, another way of creating experiment config files is to directly provide specific config groups, such as in the following example:

```yaml

defaults:
  - override /data/distribution: dirichlet.yaml
  - override /data/others: union_clients_test.yaml
  - override /save: null
  - override /eval: all.yaml

```

```{eval-rst}

.. note::
  More examples related to config groups can be found in the experiment folder ``ARMLET_DIR/configs/experiment/``.

```


2. Run `armlet` by specifying the path to the experiment config file (from the `PROJECT_DIR/configs/experiment/` folder) in the argument `+experiment=`.

```bash
armlet +experiment=template
```

Just as running experiments from command-line, multiple experiments can be run sequentially by first adding the option `-m` to the command line and then specifying multiple values (separated by a comma) for the argument `+experiment=`.

```bash
armlet -m +experiment=example/Adult/log_regression,example/ARS/svm,example/MEPS/mlp
```

Sweeping capabilities can also be directly added to the experiment config file. To do that, users need to (a) add the following lines at the end of the experiment config file and (b) provide the desired config values for the sweeping. Do not forget to add the `-m` option in the command line.

```yaml

hydra:
  sweeper:
    params:
      exp.seed: 1,2,3,4,5

```

Finally, running experiment config file can be also combined with the command line way of specifying config values.

```bash
armlet -m +experiment=template exp.seed=1,2,3,4,5
```
