(config_exp)=

# Experiment configuration

The general experiment configuration are related to how the experiment is run.
Specifically, users can define:

- `mode`: the way of running the experiment. It can be in a federated or a centralized fashion.
Note that only the `federation` mode is implemented for the moment;

- `train`: for performing the training step or not;

- `seed`: the seed for the random number generator, which is essential for the experiment **reproducibility**.
This does not apply to the data pipeline as it exists a second seed for this step;

- `device`: the device where the training and evaluation will be performed. The supported settings are `cpu`, `cuda`,`mps`, or `auto`;

- `inmemory`: whether to use caching to save memory. If `true`, the data is stored in memory, otherwise it is stored on disk.

```{eval-rst}

.. seealso::
	For more information about ``device``, ``seed``, and ``inmemory``, see `Experiment configuration <https://makgyver.github.io/fluke/config_exp.html#general-experimental-configuration>`_ from Fluke documentation.

```
