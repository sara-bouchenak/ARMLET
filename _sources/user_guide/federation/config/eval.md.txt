(config_eval)=

# Evaluation configuration

In federated learning, the evaluation can be performed in different ways.
**ARMLET** offers the same options as the `eval` config group provided by [Fluke](https://makgyver.github.io/fluke/config_exp.html#evaluation-configuration), but allows users flexibility to determine the type of evaluation.

- `_target_`: the class corresponding to the type of evaluation to perform. 
Note that only `armlet.eval.evaluators.MultiCriteriaBinaryClassEval` and `armlet.eval.evaluators.MultiClassEval` evaluators are implemented for the moment;

- `metrics`: the torch metrics that will be computed during evaluation.
Users must specify the classes of the metrics they want to evaluate in the *utility* and *fairness* sub-dictionaries.

By default, four utility metrics are selected: `torchmetrics.Accuracy`, `torchmetrics.Precision`, `torchmetrics.Recall`, and `torchmetrics.F1Score`.

Five fairness metrics are implemented in **ARMLET**: `armlet.eval.metrics.BinaryAOD`, `armlet.eval.metrics.BinaryDI`, `armlet.eval.metrics.BinaryDcI`, `armlet.eval.metrics.BinaryEOD`, and `armlet.eval.metrics.BinarySPD`.
We also provide the collection of metrics `armlet.eval.metrics.BinaryFairnessMetrics` for dynamically computing these five fairness metrics (and therefore reducing computing time).

- `eval_every`: the frequency of evaluating the models (in rounds);

- `pre_fit`: `true`, evaluation of the client model on the client side (with the client test set) before the client local training starts.
In most of the cases, this means testing the just received global model on the local test set;

- `post_fit`: `true`, evaluation of the client model on the client side (with the client test set) once the client local training has been completed.
This is useful to understand how the client model has improved during the local training;

- `server`: `true`, evaluation of the global model on the server side (with the server test set) after aggregation;

- `locals`: `true`, evaluation of the client local models on the server side (with the server test set) after local updates;

- and additional config values depending on the chosen evaluator class (that will be dynamically pass to the class as arguments).
