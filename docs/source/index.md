# **ARMLET**

<h3>Federated Learning System Multi-Criteria Benchmarking Audit</h3>

**ARMLET** is an extensible framework for multi-criteria benchmarking and audits in Federated Learning (FL).
It is designed to be modular and flexible, so that adding features is meant to be practical and simple.

## Repository Structure

```bash
├── armlet/
│   ├── audit/                      # [Coming soon] Module for performing post-hoc audit
│   ├── data/                       # Data loading pipeline (loading, splitting, cleaning, processing, ...)
│   ├── eval/                       # Evaluator for binary classification, including multi-criteria metrics
│   ├── FL_pipeline/                # Modules occuring during the FL process
│   │   ├── data_selection/         # Data selection module
│   │   └── FL_algorithms/          # Custom FL algorithms based on fluke template (client, server, ...)
│   ├── results_analysis/           # Independant module for loading and analyzing experiments results
│   └── utils/                      # Configs, loggers, losses, pytorch models
├── configs/                        # Configuation files for experiments
├── datasets/                       # Datasets
├── docs/                           # ARMLET documentation
├── outputs/                        # Experiments outputs
├── tutorials/                      # Tutorials
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Main Features

**ARMLET** can be useful if you are interested in the following features:

- We aim to provide a tool that enables multi-criteria benchmarking in FL.
With **ARMLET**, users can evaluate algorithms (e.g., ML models, FL aggregation approaches) or configurations (e.g., model hyperparameters) based on multiple criteria, such as utility, fairness, cost, or privacy.
New components, such as fairness and privacy evaluators and metrics, will be gradually integrated into this framework to extend this functionality.

- **ARMLET** can be used for performing post-hoc audit by analyzing the results of past experiments.
To do this, users need to specify metrics targets in a configuration file and run the audit tool of **ARMLET** to generate a personalized report.

- **ARMLET**'s data pipeline allows for better integrated data preprocessing, such as data normalization, data cleaning, or features encoding.
As a result, components of the data pipeline could be evaluated during benchmarking or audits.

- Due to its configuration management system, our tool makes it easy to share and relaunch experiments, ensuring robust reproductibility.
Furthermore, preconfigured configuration files simplifie the process of running experiments, making **ARMLET** easy to start with.
Our framework is also well equipped for studies needing a large number of experiments.
It provides several tools to organize experiment logs and automatically load and plot the metrics.

## Acknowledgments

**ARMLET** is based on [Fluke](https://github.com/makgyver/fluke) to start from an existing FL framework.
It serves as the basics for the FL training and evaluation processes by encompassing many FL aggregation algorithms, implementing communication between the server and clients, and providind several tools for data management.

## Explore **ARMLET**

::::{grid} 3
:::{grid-item-card} <i class="fa-solid fa-rocket"></i> Getting Started
:link: ./getting_started/install.html
Start here if it is your first time using **ARMLET**.
:::
:::{grid-item-card} <i class="fa-solid fa-code"></i> API reference
:link: ./api/modules.html
Explore the **``armlet``** API.
:::
:::{grid-item-card} <i class="fa-solid fa-laptop-code"></i> Tutorials
:link: ./getting_started/tutorials/index.html
Check out the tutorials to learn how to use **ARMLET**.
:::
::::

```{eval-rst}

.. toctree::
    :hidden:

    Getting started <getting_started/index>
    User Guide <user_guide/index>
    API reference <api/modules>
    Development <development/index>

```
