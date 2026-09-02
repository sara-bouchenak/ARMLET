(gs_run_first_exp)=

# Run your FL first experiment

**ARMLET** is based on [Hydra](https://hydra.cc/) for simplifying the launch of experiments and ensuring high reproductibily with config files.
In the following, we provide few guidelines for running your first experiment.

```{eval-rst}

.. seealso::
	Please look at `Hydra documentation <https://hydra.cc/docs/intro/>`_ for further comprehension about Hydra basics.

```

Run `armlet` to launch a single FL experiment with the default configuration values.

```bash
armlet
```

```{eval-rst}

.. warning::
	If you are in ``PROJECT_DIR``, you must (a) create a folder ``datasets`` and add your datasets or (b) change the ``paths.data_dir`` config value.
	For running default configuration fastly without error, just copy and paste ``ARMLET_DIR/datasets`` to ``PROJECT_DIR``.

```

Config values (e.g., `exp.seed`) or config groups (e.g., `data/dataset`) can be directly overrided from the command line.
Note that `+` need to be added before the config values or groups that do not have default values.

```bash
armlet exp.seed=1 data/dataset=tabular/ars +data/saving=after_processing
```

```{eval-rst}

.. seealso::
	For more information about configuration files, config values, or config groups, see :ref:`Federation configuration <config_federation_mode>`.

```

```{eval-rst}

.. tip::
	You can also run multiple experiments sequentially in a single command-line or use experiment config files.
	For more information, see :ref:`Running FL experiments <federation_run>`.

```
