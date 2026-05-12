(config)=

# Configuration prerequisites

In this page, we provide an overview of the general configuration groups and values that are used in each mode of **ARMLET**.
This way of managing configurations is based on the one proposed by [Fluke](https://makgyver.github.io/fluke/configuration.html) and relies on YAML.
Nevertheless, we make several improvements to offer greater flexibility in managing experiments.

```{eval-rst}

.. important::
  For the following or any other configuration explanations, we distinguish between two types of configuration elements:
  the **configuration values** (e.g., ``exp.seed``), which can be accessed using a ``.``,
  and the **configuration groups** (e.g., ``data/dataset``), which refer to a configuration file containing one or more config values and must be accessed using a ``/``.

```

In **ARMLET**, we compose both config groups and config values to run experiments with the desired configurations.

## General configuration categories

The general configuration groups used in each **ARMLET** mode are:

- [`armlet`](config_armlet): for choosing the mode of **ARMLET**;

- [`experiment`](config_experiment): for composing multiple configurations;

- [`hydra`](config_hydra): for managing Hydra;

- [`paths`](config_paths): for the general paths (data, log, output).

```{eval-rst}

.. seealso::
  All the essential config values are explained in the subpages, but the different options (i.e., config files) for each config group are not detailed.
  Please look at the ``ARMLET_DIR/configs`` folder to explore the different config group possibilities.

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    Armlet<armlet>
    Experiment<experiment>
    Hydra<hydra>
    Paths<paths>

```
