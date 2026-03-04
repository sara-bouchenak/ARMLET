# ARMLET

<h3>Federated Learning System Multi-Criteria Benchmarking Audit</h3>

**ARMLET** is an extensible framework for multi-criteria benchmarking and audits in Federated Learning (FL).
It is designed to be modular and flexible, so that adding features is meant to be practical and simple.

## Table of Contents

- [Main Features](#main-features)
- [How to install ARMLET](#how-to-install-armlet)
- [Run your first FL experiment](#run-your-first-fl-experiment)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Authors and main contributors](#authors-and-main-contributors)
- [Acknowledgments](#acknowledgments)

## Main Features

**ARMLET** can be useful if you are interested in the following features:

- We aim to provide a tool that enables multi-criteria benchmarking in FL.
With **ARMLET**, users can evaluate algorithms (e.g., ML models, FL aggregation approaches) or configurations (e.g., model hyperparameters) based on multiple criteria, such as utility, fairness, cost, or privacy.
New components, such as fairness and privacy evaluators and metrics, will be gradually integrated into this framework to extend this functionality.

- **ARMLET** can be used for performing audit by analyzing the results of past experiments.
To do this, users need to specify metrics targets in a configuration file and run the audit tool of **ARMLET** to generate a personalized report.

- **ARMLET**'s data pipeline allows for better integrated data preprocessing, such as data normalization, data cleaning, or features encoding.
As a result, components of the data pipeline could be evaluated during benchmarking or audits.

- Due to its configuration management system, our tool makes it easy to share and relaunch experiments, ensuring robust reproductibility.
Furthermore, preconfigured configuration files simplifie the process of running experiments, making **ARMLET** easy to start with.
Our framework is also well equipped for studies needing a large number of experiments.
It provides several tools to organize experiment logs and automatically load and plot the metrics.

## How to install ARMLET

First, clone the **ARMLET** repository:

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

To use your own configurations, see the following documentation page: [Running experiments](https://sara-bouchenak.github.io/ARMLET/user_guide/run_exp.html).

## Documentation

The documentation for **ARMLET** can be found [here](https://sara-bouchenak.github.io/ARMLET/).
You will find detailed information on how the package works, how to install and use it, tutorials, and precisions about **ARMLET** configurations.

## Tutorials

[IN PROGRESS]

## Authors and main contributors

- [Baudouin Naline](https://github.com/bnaline), LIRIS Lab, CNRS & INSA Lyon - *Idealization*, *Design*, *Development*, and *Documentation*
- [James Sudlow](https://github.com/JamesSudlow), LIRIS Lab, CNRS & INSA Lyon - *Development*
- Artur Vieira Pereira, LIRIS Lab, CNRS & INSA Lyon - *Development*

## Acknowledgments

**ARMLET** is based on [Fluke](https://github.com/makgyver/fluke) to start from an existing FL framework.
It serves as the basics for the FL training and evaluation processes by encompassing many FL aggregation algorithms, implementing communication between the server and clients, and providind several tools for data management.
