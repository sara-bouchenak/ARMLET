(user_guide)=

# User Guide

**ARMLET** provides different modes that can be accessed with the argument `--config-name=MODE_NAME` (for running the default configurations related to the chosen mode) or the config value `armlet.mode=MODE_NAME` (in this case, some config values should be specified manually). There are:

- [`federation`](federation_mode): to run federated learning experiments;

- [`audit`](audit_mode): to perform post-hoc audit of past FL experiments;

- and other modes that are currently in development and will be available soon.

On these subpages, you will find more detailed information on how to use these modes.
Furthermore, we explain all the configurations of **ARMLET** and add information on how to extend our framework.

```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    config/index
    federation/index
    audit/index
    extend

```
