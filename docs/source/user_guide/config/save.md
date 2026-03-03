(config_save)=

# Saving configuration

This configuration group is exactly the same as the `save` config group defined in [Fluke](https://makgyver.github.io/fluke/config_exp.html#saving-configuration).

It allows to specify where to save the models and how often to save them.
It is an optional section and can be omitted if the user does not want to save the models.
To save the models, the user must specify the following parameters:

- `path`: the path to the folder where to save the models;

- `save_every`: the frequency of saving the models (in rounds);

- `global_only`: whether to save only the global model.

```{eval-rst}

.. seealso::
	For more information about the naming convention when saving models, see `Experiment configuration <https://makgyver.github.io/fluke/config_exp.html#saving-configuration>`_ from Fluke documentation.

```
