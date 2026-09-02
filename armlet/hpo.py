import os
import shutil
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from armlet.utils.configs import ArmletConfiguration
from armlet.federation import run_federation
from armlet.audit.load_metrics import load_df_run, preprocess_df_metrics


def run_hpo_federation(cfg: DictConfig) -> None:

    cfg.paths.root_dir = os.getcwd()
    cfg_copy = cfg.copy()
    OmegaConf.resolve(cfg_copy)
    hpo_dir = cfg_copy.paths.output_dir

    objective_partial = partial(
        objective,
        cfg=cfg.copy(),
        hpo_dir=hpo_dir,
    )

    objective_with_resources = tune.with_resources(
        objective_partial,
        {
            "cpu": cfg.hpo.n_cpu_per_task,
            "gpu": cfg.hpo.n_gpu_per_task,
        },
    )

    search_space = hydra.utils.call({"_target_": cfg.hpo.search_space_func})

    optuna_sampler = hydra.utils.instantiate(cfg.hpo.optuna_sampler)
    optuna_search = OptunaSearch(sampler=optuna_sampler)

    tuner = tune.Tuner(
        objective_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            search_alg=optuna_search,
            num_samples=cfg.hpo.n_trials,
        ),
        run_config=tune.RunConfig(
            storage_path=hpo_dir,
            name="ray_results",
        ),
    )

    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

    best_config = results.get_best_result().config
    best_config_path = os.path.join(hpo_dir, "best_cfg_method.yaml")
    with open(best_config_path, 'w') as config_file:
        OmegaConf.save(config=best_config, f=config_file.name)

    df_results = results.get_dataframe()
    df_results_path = os.path.join(hpo_dir, "hpo_summary.csv")
    df_results.to_csv(df_results_path)

def objective(ray_config, cfg, hpo_dir):

    gpu_id = ray.get_gpu_ids()[0]
    trial_dir = os.path.join(hpo_dir, f"trial_gpu_{gpu_id}")
    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir)
    os.mkdir(trial_dir)
    cfg.paths.output_dir = trial_dir
    OmegaConf.resolve(cfg)
    trial_cfg = merge_dicts(cfg.copy(), ray_config)

    # run armlet federation pipeline 
    trial_cfg = ArmletConfiguration(trial_cfg)
    run_federation(trial_cfg)

    # Load results and compute score
    result_path = os.path.join(trial_dir, "results.json")
    df_results = load_df_run(result_path, ["perf_global"])
    df_results = preprocess_df_metrics(df_results, no_metric_cat=True)
    metrics = hydra.utils.call(
        cfg.hpo.score_func,
        df_results=df_results,
    )

    shutil.rmtree(trial_dir)
    return metrics

def merge_dicts(hydra_cfg, ray_cfg):
    if not isinstance(hydra_cfg, dict) and not isinstance(hydra_cfg, DictConfig):
        return ray_cfg
    for key in hydra_cfg.keys():
        if key in ray_cfg.keys():
            hydra_cfg[key] = merge_dicts(hydra_cfg[key], ray_cfg[key])
    return hydra_cfg

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
