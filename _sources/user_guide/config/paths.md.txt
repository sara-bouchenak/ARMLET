(config_paths)=

# Paths configuration

The `paths` config group is used to configure the paths.

- `root_dir`: path to root directory;

- `data_dir`: path to data directory;

- `log_dir`: path to logging directory;

- `output_dir`: path to output directory. By default, the path is created dynamically by Hydra with the value `${hydra:runtime.output_dir}`.
In this case, the path generation pattern is specified in the [`hydra`](config_hydra) config group.
