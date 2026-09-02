(config_hpo)=

# HPO configuration

The `hpo` mode of **ARMLET** uses all the configurations of the federation mode (see [Federation configuration](config_federation_mode)) and requires the `hpo` config group to define the hyperparameter optimization settings. This specific group includes:

- `n_trials`: number of times to sample from the hyperparameter space, which corresponds to the number of search iterations;

- `n_gpu_per_task`: number of GPUs required for a trial. This allows tasks to be parallelized if the machines have multiple ressources;

- `n_cpu_per_task`: number of CPUs required for a trial;

- `search_space_func`: function used to define the search space;

- `score_func`: config group that defines how to compute the score to be optimized during the HPO process. It must contain:
    - `_target_`: the function;
    - the arguments of the function if needed;

- `optuna_sampler`: config group that defines which [Optuna sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) will be used by Ray Tune. It must contain:
    - `_target_`: the Optuna sampler class;
    - the arguments of the class if needed.

Here is an example of the `hpo` config group:

```yaml

n_trials: 100
n_gpu_per_task: 1
n_cpu_per_task: 1

search_space_func: armlet.hpo.compute_search_space

score_func:
  _target_: armlet.hpo.compute_score
  metric_to_optimize: accuracy

optuna_sampler:
  _target_: optuna.samplers.TPESampler
  seed: 42
  n_startup_trials: 10

```
