(config_logger)=

# Logger configuration

The `logger` config group is used to specify the logging class to be used during the experiment.
Unlike in [Fluke](https://makgyver.github.io/fluke/config_log.html), the logging class must be specified in the `_target_` config value.

- `_target_`: the class corresponding to the type of logging to perform.

**ARMLET** provides by default a logger, named `armlet.utils.log.ArmletLog`, which is in charge of calculating several communication and computational costs. At the end of the experiment, it saves the results in a JSON file.

Regarding the communication costs, **ARMLET** can track the model size (in bits) and the number of model parameters, either for the server or each client.
To activate this functionnality, users must add the config group `comm_costs_tracker` with the config values:

- `track_every`: the tracking frequency (in rounds);

- `clients_at_end_fit`: whether to calculate these metrics for each client (boolean);

- `server_at_start_round`: whether to calculate these metrics for the server (boolean).

In the following, you can find an example that describes the default settings (`armlet/configs/logger/default.yaml` config file):

```yaml

comm_costs_tracker:
  track_every: ${protocol.n_rounds}
  clients_at_end_fit: true
  server_at_start_round: true

```

Moreover, **ARMLET** can calculate the time needed to reach target metrics.
This feature can be enabled by adding the config group `metric_target` with as many config subgroups as there are metrics to monitor.
The config subgroups must include:

- `threshold`: the threshold to reach for the corresponding metric;

- `objective`: whether the metric value must be `above` or `below` the threshold.

Here is an example of such config group (as in the `armlet/configs/logger/time_to_reach_target.yaml` config file):

```yaml

metric_target:
  accuracy:
    threshold: 0.8
    objective: above
  f1:
    threshold: 0.8
    objective: above

```
