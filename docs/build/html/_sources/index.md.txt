# **Armlet**

<h3>Federated learning system multi-criteria benchmarking audit.</h3>

**Armlet** is an extensible framework for multi-criteria benchmarking and audits in Federated Learning (FL).
It is designed to be modular and flexible, so that adding features is meant to be practical and simple.

## Repository Structure

```bash
├── armlet/
│   ├── data/                       # Data loading pipeline (loading, splitting, cleaning, processing, ...)
│   ├── eval/                       # Evaluator for binary classification, including utility and fairness metrics
│   ├── FL_pipeline/                # Modules occuring during the FL process
│   │   ├── data_selection/         # Data selection module
│   │   └── FL_algorithms/          # Custom FL algorithms based on fluke template (client, server, ...)
│   ├── results_analysis/           # Independant module for loading and analyzing experiments results
│   └── utils/                      # Configs, loggers, losses, pytorch models
├── configs/                        # Configuation files for experiments
├── datasets/                       # Datasets
├── docs/                           # Armlet documentation
├── outputs/                        # Experiments outputs
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Main Features

**Armlet** can be useful if you are interested in the following features:

- We aim to provide a tool that enables multi-criteria benchmarking and audit in FL.
To achieve it, our tool starts from the existing [Fluke](https://github.com/makgyver/fluke) framework and plans to integrate new components, such as fairness and privacy evaluators and metrics.

- **Armlet**'s data pipeline allows for better integrated data preprocessing before using Fluke. As a result, components of the data pipeline could be evaluated during benchmarking or audits.

- Due to its configuration management system using Hydra, our tool makes it easy to share and relaunch experiments, ensuring robust reproductibility.
Furthermore, preconfigured configuration files simplifie the process of running experiments, making **Armlet** easy to start with.

- Our framework is well equipped for studies needing a large number of experiments. It provides several tools to organize experiment logs and automatically load and plot the metrics.

## Acknowledgments

**Armlet** is based on [Fluke](https://github.com/makgyver/fluke) to start from an existing FL framework and add fairness-specific and data-processing components to it.

## Explore **Armlet**

::::{grid} 3
:::{grid-item-card} <i class="fa-solid fa-rocket"></i> Getting Started
:link: ./getting_started/install.html
Is it your first time using **Armlet**? Start here.
:::
:::{grid-item-card} <i class="fa-solid fa-code"></i> API reference
:link: ./api/modules.html
Explore the **``armlet``** API.
:::
:::{grid-item-card} <i class="fa-solid fa-laptop-code"></i> Tutorials
:link: ./getting_started/tutorials/index.html
Check out the tutorials to get acquainted with **Armlet**.
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
