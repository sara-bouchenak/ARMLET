(user_guide)=

# User Guide

**ARMLET** offers several services, which can be used by selecting the proper mode using the argument `--config-name=MODE_NAME` or `-cn MODE_NAME`.
The different modes are:

- [`federation`](federation_run): to run federated learning experiments;

- [`hpo`](hpo_run): to tune any parameters of the federation mode using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html);

- [`post_hoc_audit`](audit_run_post_hoc): to perform post-hoc audit of past FL experiments;

- and other modes that are currently in development and will be available soon.

**ARMLET** also includes [`plot`](ug_plot_module), an independant module for analyzing and plotting experiment results.

On these subpages, you will find more detailed information on how to use these modes or modules.
Furthermore, we explain all the configurations of **ARMLET** and add information on [how to extend our framework](extend).

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Benchmark & FL experiments

    federation/run
    federation/config/index
    federation/metrics
    federation/datasets
    federation/models
    federation/features/index
    federation/tutorials/index

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Audit

    audit/index
    audit/post_hoc_audit
    audit/online_audit
    audit/config

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Hyperparameters Optimization

    hpo/run
    hpo/config

```

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Others

    plot_module
    extend

```
