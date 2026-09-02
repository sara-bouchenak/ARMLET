(feature_resume_training)=

# Resume training

An FL experiment can be resumed from a previously saved state.

Users have first to save the models by using the `save` config group (see [Saving configuration](config_save)), as for instance by running:
```bash
armlet save=every_round
```
The models will be hence saved in the `save.path` directory.

```{eval-rst}

.. important::
	The config value ``save.global_only`` must be set to ``false`` for allowing resume training.

```

Then, the experiment can be resumed by specifying the config value `method.resume.path`, such as:

```bash
armlet +method.resume.path=TORCH_MODELS_DIR
```
where `TORCH_MODELS_DIR` is the directory containing the torch models (located in the `save.path` directory of the previous experiment).

By default, **ARMLET** loads the models of the last rounds located in the `TORCH_MODELS_DIR` directory.
For loading other models, users can add the config value `method.resume.round`, such as:
```bash
armlet +method.resume.path=TORCH_MODELS_DIR +method.resume.round=ROUND
```
