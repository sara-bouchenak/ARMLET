(config_hydra)=

# Hydra configuration

The `hydra` configuration is specific to the operation of Hydra.
Users can customize the Hydra config values by following this tutorial: [Configuring Hydra](https://hydra.cc/docs/configure_hydra/intro/).

**ARMLET** provides a default hydra config file that configures the path generation pattern of the `output_dir` config value:

```yaml

run:
  dir: ${paths.root_dir}/outputs/default/${now:%Y-%m-%d}_${now:%H-%M-%S}/dataset=${data.dataset.dataset_name}

sweep:
  dir: ${paths.root_dir}/outputs/default/${now:%Y-%m-%d}_${now:%H-%M-%S}/dataset=${data.dataset.dataset_name}
  subdir: ${hydra.job.num}_${sanitize_override_dirname:${hydra.job.override_dirname}}

job:
  config:
    override_dirname:
      exclude_keys:
        - experiment

```
