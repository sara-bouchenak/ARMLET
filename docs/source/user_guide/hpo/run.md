(hpo_run)=

# Tuning hyperparameters

**ARMLET** can automatically tune the hyperparameters of the federation mode by using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).
For this purpose, **ARMLET** provides the HPO mode, which offers an interface between Ray Tune and the federation mode.

## How to run ?

The HPO mode can be used by running the following command:

```bash
armlet -cn hpo
```

```{eval-rst}

.. important::
	All Hydra guidelines explained in :ref:`Running FL experiments <federation_run>` also apply for this mode, allowing users to override the configurations directly from the command line or using a YAML experiment config file.

```

## How does it work ?

This command will:

1. initialize the configuration values of the federation mode (see [Running FL experiments](federation_run)) that Ray Tune will start;

2. set up the hyperparameters optimization process according to the HPO settings. More precisely, it will:
    1. define the objective we want to tune (i.e., prepare the trial configs by overriding the federation configs with the values proposed by the sampler; run the federation mode of **ARMLET** with the trial configs; retrieve the metrics; and compute a score by means of the **score function**);
    2. specify the ressources (CPUs and GPUs) needed for a tuning iteration;
    3. define the hyperparameters we want to tune by means of the **search space function**;
    4. instantiate the Optuna sampler and the Tune’s search algorithm;
    5. instantiate the `Tuner` object by using the previous elements and precising the number of trials;

3. call `Tuner.fit()` to execute and manage hyperparameter tuning and generate the trials. For each trial, Ray Tune samples a combinaison of HPs (by using the search space function and the Optuna sampler) and runs the objective function.

4. Once hyperparameter tuning is complete, the best combinaison of HPs are saved in the `OUTPUT_DIR` directory along with a summary of the process.

## Search space and score functions

**ARMLET** provides a default functions for defining the search space (i.e., `armlet.hpo.compute_search_space()`) and computing the score (i.e., `armlet.hpo.compute_score()`):

```python

def compute_search_space():
    search_space = {
        "method": {
            "hyperparameters": {
                "client": {
                    "optimizer": {
                        "name": tune.choice(["SGD", "Adam", "Adagrad", "AdamW", "RMSprop"]),
                        "lr": tune.loguniform(1e-5, 1e-1),
                        "weight_decay": tune.choice([0.0, 1e-4, 1e-2]),
                    },
                    "batch_size": tune.choice([64, 128, 256, 1024]),
                },
            },
        },
    }
    return search_space

```

```python

def compute_score(df_results, metric_to_optimize):
    if metric_to_optimize in df_results["metric_name"].unique():
        df_metric = df_results[df_results["metric_name"] == metric_to_optimize]
        df_last_10_rounds = df_metric.loc[df_metric["round"] > df_metric["round"].max() - 10]
        avg_last_10_rounds = df_last_10_rounds["metric_value"].mean()
        std_last_10_rounds = df_last_10_rounds["metric_value"].std()
        avg_overall = df_metric["metric_value"].mean()
        score = 0.7 * avg_last_10_rounds + 0.3 * avg_overall - std_last_10_rounds
    else:
        raise ValueError("The metric to optimize is not correct!")
    return {
        "score": score,
        "avg_overall": avg_overall,
        "avg_last_10_rounds": avg_last_10_rounds,
        "std_last_10_rounds": std_last_10_rounds, 
    }

```

Since users have varying needs and work on different use cases, **ARMLET** offers the possibility to implement custom search space and score functions, which can be specified in the `hpo` config groups.

## HPO outputs

[TODO]
