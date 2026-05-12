(config_method)=

# Method configuration

In this page, we detail the configuration values related to the FL algorithm and its hyperparameters.
The configuration groups or values in `method` are:

- `_target_`: the class corresponding to the FL algorithm to apply;

- `hyperparameters`: the hyperparameters of the FL algorithm, across different config groups;

    - [`client`](config_method_client): for the client hyperparameters;

    - [`server`](config_method_server): for the server hyperparameters;

    - [`model`](config_method_model): for the model hyperparameters;

    - and additional config groups depending on the chosen FL class.

## Client hyperparameters
(config_method_client)=

The `client` field includes all the hyperparameters that are related to the client-side of the federated algorithm.
It contains:

- `batch_size`: batch size;

- `local_epochs`: number of local epochs performed during each FL round;

- `loss`: config group that contains the information about the loss function. Users should specify the `_target_` config value, which represents the loss function to instantiate (e.g., `torch.nn.BCELoss`), with the required additional parameters;

- `optimizer`: config group that contains the hyperparameters related to the optimizer. Users should specify the `_target_` config value, which represents the optimizer to instantiate (e.g., `Adam`), with the required additional parameters (e.g., the learning rate or the weight decay);

- `scheduler`: config group that contains the hyperparameters related to the scheduler. Users should specify the `_target_` config value, which represents the scheduler to instantiate (e.g., `StepLR`), with the required additional parameters (e.g., the gamma or the step size);

- and additional config values or groups depending on the chosen FL algorithm.

## Server hyperparameters
(config_method_server)=

The `server` field includes all the hyperparameters that are related to the server-side of the federated algorithm.
It contains:

- `loss`: config group that contains the information about the loss function (the same as the client loss config group), which will be used only for the server evaluation;

- `time_to_accuracy_target`: value between 0 and 1 representing the accuracy target to achieve (used for calculating time to accuracy);

- `weighted`: boolean specifying whether to weight the client’s contribution to the global model;

- and additional config values or groups depending on the chosen FL algorithm.

## Model hyperparameters
(config_method_model)=

The `model` field specifies the machine learning model to federate.
It contains:

- `_target_`: the class corresponding to the ML model to instantiate (e.g., `armlet.utils.net.LogRegression`);

- `input_size`: the input size of the ML model (**automatically loaded in ARMLET** according to the data properties);

- `num_classes`: the number of output classes of the ML model (**automatically loaded in ARMLET** according to the data properties);

- and additional config values or groups depending on the chosen ML model.

Note that an ML model is usually linked to some hyperparameters located in the `client` and `server` config groups (such as the loss function).
Consequently, the preconfigured options for the `model` config group in **ARMLET** could override other config values that are originally included in the `client` and `server` config groups.
