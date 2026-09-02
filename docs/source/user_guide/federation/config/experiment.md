(config_experiment)=

# Experiment configuration

The `experiment` configuration is specific to the operation of Hydra.
It aims to cleanly support multiple configurations, with each configuration file only specifying the changes to the default configuration.
Thus, we encourage users to create their config files in the  ``PROJECT_DIR/configs_armlet/experiment`` folder
by specifying the overrides to the default configuration, and then call it via the command line:

```bash
armlet +experiment=my_experiment
```


```{eval-rst}

.. seealso::
    For more information, see `Configuring Experiments <https://hydra.cc/docs/patterns/configuring_experiments/>`_ from Hydra documentation.

```
