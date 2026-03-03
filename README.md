# Armlet

<h3>Federated learning system multi-criteria benchmarking audit.</h3>

**Armlet** is an extensible framework for multi-criteria benchmarking and audits in Federated Learning (FL).
It is designed to be modular and flexible, so that adding features is meant to be practical and simple.

## Table of Contents

- [Main Features](#main-features)
- [How to install Armlet](#how-to-install-armlet)
- [Run your first FL experiment](#run-your-first-fl-experiment)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Authors and main contributors](#authors-and-main-contributors)
- [Acknowledgments](#acknowledgments)

## Main Features

**Armlet** can be useful if you are interested in the following features:

- We aim to provide a tool that enables multi-criteria benchmarking and audit in FL.
To achieve it, our tool starts from the existing [Fluke](https://github.com/makgyver/fluke) framework and plans to integrate new components, such as fairness and privacy evaluators and metrics.

- **Armlet**'s data pipeline allows for better integrated data preprocessing before using Fluke. As a result, components of the data pipeline could be evaluated during benchmarking or audits.

- Due to its configuration management system using Hydra, our tool makes it easy to share and relaunch experiments, ensuring robust reproductibility.
Furthermore, preconfigured configuration files simplifie the process of running experiments, making **Armlet** easy to start with.

- Our framework is well equipped for studies needing a large number of experiments. It provides several tools to organize experiment logs and automatically load and plot the metrics.

## How to install Armlet

First, clone the **Armlet** repository:

```bash
git clone https://github.com/sara-bouchenak/ARMLET.git
```

Then, install **``armlet``** using `pip`:

```bash
cd path/to/armlet/project
pip install .
```

## Run your first FL experiment

First, go to your project directory:

```bash
cd path/to/your/project
```

Then, run `armlet` to launch a single experiment with the default configuration values.

```bash
armlet
```

To use your own configurations, see the following documentation page: [Running experiments]().

## Documentation

The documentation for **Armlet** can be found [here]().
You will find detailed information on how the package works, how to install and use it, tutorials, and precisions about **Armlet** configurations.

## Tutorials

[IN PROGRESS]

## Authors and main contributors

- [Baudouin Naline](https://github.com/bnaline), LIRIS Lab, CNRS & INSA Lyon - *Idealization*, *Design*, *Development*, and *Documentation*
- [James Sudlow](https://github.com/JamesSudlow), LIRIS Lab, CNRS & INSA Lyon - *Development*
- Artur Vieira Pereira, LIRIS Lab, CNRS & INSA Lyon - *Development*

## Acknowledgments

**Armlet** is based on [Fluke](https://github.com/makgyver/fluke) to start from an existing FL framework and add fairness-specific and data-processing components to it.
