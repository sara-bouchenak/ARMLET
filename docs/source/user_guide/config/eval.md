(config_eval)=

# Evaluation configuration

In federated learning, the evaluation can be performed in different ways.
**ARMLET** offers the same options as the `eval` config group provided by [Fluke](https://makgyver.github.io/fluke/config_exp.html#evaluation-configuration), but allows users flexibility to determine the type of evaluation.

- `_target_`: the class corresponding to the type of evaluation to perform. 
Note that only `armlet.eval.evaluators.MultiCriteriaBinaryClassEval` evaluator is implemented for the moment;

- `eval_every`: the frequency of evaluating the models (in rounds);

- `pre_fit`: `true`, evaluation of the client model on the client side (with the client test set) before the client local training starts.
In most of the cases, this means testing the just received global model on the local test set;

- `post_fit`: `true`, evaluation of the client model on the client side (with the client test set) once the client local training has been completed.
This is useful to understand how the client model has improved during the local training;

- `server`: `true`, evaluation of the global model on the server side (with the server test set) after aggregation;

- `locals`: `true`, evaluation of the client local models on the server side (with the server test set) after local updates;

- and extra parameters depending on the chosen evaluator class.
