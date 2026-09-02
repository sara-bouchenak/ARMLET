(config_audit)=

# Audit configuration

The `post_hoc_audit` mode or the online audit feature of the `federation` mode include  the configurations specific to auditing into the `audit` category.
There are:

- `exp_dir`: **only for post-hoc-mode**, path to the folder containing the collection of experiments to audit;

- `metric_types_to_audit`: config group that contains booleans. They enable or disable the audit for specific metric types (`perf_global`, `perf_locals`, `perf_prefit`, `perf_postfit`, `comm_costs`, or `comp_costs`);

- `last_n_rounds`: the default value for filtering the metrics associated to a single time series (it filters the metrics of the *n* last rounds of each experiment);

- `agg_func`: the default function for aggregating the metrics associated to a single time series (e.g., `mean`, `min`, `max`, `sum`). It can also be set to `null` to prevent the metrics from being aggregated;

- as many config groups as metrics to audit. These config groups must be organized by metric categories (the key must be in the form `{CATEGORY_NAME}_metrics`, where `{CATEGORY_NAME}` corresponds to the category name, e.g., *utility* or *fairness*). Such config group includes:
    - `objective`: whether the aggregated metric value (or multiple metric values if `agg_func` is set to `null`) must be `above` or `below` the thresholds;
    - `threshold_good`: the threshold that must be reached to achieve the audit result *good*;
    - `threshold_medium`: the threshold that must be reached to achieve the audit result *medium*;
    - `last_n_rounds`: **optional**, the value that replaces the default `audit.last_n_rounds` value;
    - `agg_func`: **optional**, the aggregation function that replaces the default `audit.agg_func` value.

```{eval-rst}

.. important::
  When running the ``post_hoc_audit`` mode, **ARMLET** also requires the configuration groups ``armlet``, ``hydra``, ``paths``, which are detailed in :ref:`Federation configuration <config_federation_mode>`.

```

Here is an example showing how to organize these configuration groups:

```yaml

_target_: armlet.audit.post_hoc_audit_model_from_results_json

exp_dir: ${paths.root_dir}/outputs/example/post_hoc_audit

metric_types_to_audit:
  perf_global: True
  perf_locals: True
  perf_prefit: True
  perf_postfit: True
  comp_costs: True
  comm_costs: True

last_n_rounds: 20
agg_func: mean

utility_metrics:

  accuracy:
    objective: above
    threshold_medium: 82
    threshold_good: 83
    last_n_rounds: 10
    agg_func: null

  recall:
    objective: above
    threshold_medium: 80
    threshold_good: 82

fairness_metrics:

  age_eod:
    objective: below
    threshold_medium: 10
    threshold_good: 5

  gender_eod:
    objective: below
    threshold_medium: 10
    threshold_good: 5

cost_metrics:

  training_time:
    objective: below
    threshold_medium: 350
    threshold_good: 300
    last_n_rounds: 1
    agg_func: null

  tot_model_size_in_bits:
    objective: below
    threshold_medium: 12000
    threshold_good: 10000
    last_n_rounds: 1
    agg_func: null

```
