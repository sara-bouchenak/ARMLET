(config_logger)=

# Logger configuration

The `logger` config group is used to specify the logging class to be used during the experiment.
Unlike in [Fluke](https://makgyver.github.io/fluke/config_log.html), the logging class must be specified in the `_target_` config value.

- `_target_`: the class corresponding to the type of logging to perform.

**ARMLET** provides by default a minimal logger, named `armlet.utils.log.ArmletLog`, which saves the results in a JSON file.

```{eval-rst}

.. seealso::
	In `Log configuration <https://makgyver.github.io/fluke/config_log.html>`_ from Fluke documentation, you can find the loggers provided by Fluke, such as `Local logger`, `Debug logger`, `Weights & Biases logger`, or `Tensorboard logger`.
	You can also find some guidelines for defining custom loggers.

```
