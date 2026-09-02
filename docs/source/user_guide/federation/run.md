(federation_run)=

# Running FL experiments

The first use of **ARMLET** is to run federated learning experiments.
This is done through the federation mode, which is the mode chosen by default when running the `armlet` command.
**ARMLET** is based on [Hydra](https://hydra.cc/) for simplifying the launch of experiments and ensuring high reproductibily with config files.
In the following, we provide examples that illustrate how to efficiently run FL experiments with our framework,
via simple command-line (A) or through experiment config files (B).

```{eval-rst}

.. seealso::
	Please look at `Hydra documentation <https://hydra.cc/docs/intro/>`_ for further comprehension about Hydra basics.

```

## How to run ?

### A- Run FL experiment(s) from command-line

Run `armlet` to launch a single FL experiment with the default configuration values.

```bash
armlet
```

```{eval-rst}

.. warning::
	If you are in ``PROJECT_DIR``, you must (a) create a folder ``datasets`` and add your datasets or (b) change the ``paths.data_dir`` config value.
	For running default configuration fastly without error, just copy and paste ``ARMLET_DIR/datasets`` to ``PROJECT_DIR``.

```

Config values (e.g., `exp.seed`) or config groups (e.g., `data/dataset`) can be directly overrided from the command line.
Note that `+` need to be added before the config values or groups that do not have default values.

```bash
armlet exp.seed=1 data/dataset=tabular/ars +data/saving=after_processing
```

```{eval-rst}

.. seealso::
	For more information about configuration files, config values, or config groups see :ref:`Federation configuration <config_federation_mode>`.

```

Multiple FL experiments can also be run sequentially by first adding the option `-m` to the command line and then providing multiple values (separated by a comma) for the config values or the config groups.
In this case, all combinations of values to be sweeped will be crossed to launch a batch of experiments.

```bash
armlet -m exp.seed=1,2,3,4,5 data/dataset=tabular/ars,tabular/heart
```

### B- Run FL experiment(s) from a YAML experiment config file

1. Create a YAML configuration file in `PROJECT_DIR/configs_armlet/experiment/` by copying, pasting, and overriding `ARMLET_DIR/armlet/configs/experiment/template.yaml`.

```{eval-rst}

.. tip::
  Config files saved during a past experiment (``EXP_PATH/config.yaml``) can notably be copied and pasted to ``PROJECT_DIR/configs_armlet/experiment/`` to reproduce the experiment.
  In that case, add ``# @package _global_`` at the beggining of the file just as in the following example.

```

**YAML configuration file example:**

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

.. important::
  Just as from command-line, config values that are specified in the experiment config file override the default configuration.
  Thus, config values that are not provided in this file will be set to their default values.

```

For greater efficiency, another way of creating experiment config files is to directly provide specific config groups, such as in the following example:

```yaml
# @package _global_

defaults:
  - override /data/splitter: union_clients_test.yaml
  - override /data/splitter/distribution: dirichlet.yaml
  - override /save: no_model_saving.yaml
  - override /eval: binary_metrics.yaml

```

```{eval-rst}

.. note::
  More examples related to config groups can be found in the experiment folder ``ARMLET_DIR/armlet/configs/experiment/examples``.

```


2. Run `armlet` by specifying the path to the experiment config file (starting from the `PROJECT_DIR/configs_armlet/experiment/` folder) in the argument `+experiment=`.

```bash
armlet +experiment=template
```

Just as running experiments from command-line, multiple experiments can be run sequentially by first adding the option `-m` to the command line and then specifying multiple values (separated by a comma) for the argument `+experiment=`.

```bash
armlet -m +experiment=examples/benchmark/ARS/mlp,examples/benchmark/Adult/svm
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

## How does it work ?

A detailed explanation of the federation mode of **ARMLET** is presented in the tutorial [Understanding the **federation mode** in ``armlet``](https://sara-bouchenak.github.io/ARMLET/user_guide/federation/tutorials/federation_mode.html).

## FL experiment outputs

[TODO]
