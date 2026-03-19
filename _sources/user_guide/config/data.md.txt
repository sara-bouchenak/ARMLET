(config_data)=

# Data configuration

In this page, we detail the configuration values related to the data pipeline.
The configuration groups or values in `data` are:

- [`dataset`](config_data_dataset): for loading the dataset and transforming it to the appropriate format;

- [`distribution`](config_data_distribution): for specifying the data distribution when partitionning;

- [`others`](config_data_others): for the others parameters used to split the data among the clients and the server;

- `seed`: the seed for the random number generator in the data pipeline, which is essential for the experiment **reproducibility**;

- [`processing`](config_data_processing): **optional**, for performing data processing;

- [`cleaning`](config_data_cleaning): **optional**, for performing data cleaning;

- [`loading`](config_data_loading): **optional**, for dynamically loading the splitted dataset that were saved during previous experiments;

- [`saving`](config_data_saving): **optional**, for saving data at one or multiple specific steps of the data pipeline.

## Dataset configuration
(config_data_dataset)=

The `dataset` field requires two mandatory config values:

- `dataset_name`: the name of the dataset, used by the logger;

- `_target_`: the function used to load the dataset.

Other config values can be specified according to the chosen loading function.

```{eval-rst}

.. important::
  By default, **ARMLET** provides multiple dataset loading functions in ``armlet.data.datasets`` and pre-defined dataset YAML config files in the ``ARMLET_DIR/configs/data/dataset`` folder.

```

## Data distribution
(config_data_distribution)=

The `distribution` field is similar to the one provided by [Fluke](https://makgyver.github.io/fluke/config_data.html#data-distribution), but is adapted to the data format required by **ARMLET**.
The only mandatory config value is `_target_`, which is the distribution function used during the data splitting step.

```{eval-rst}

.. important::
  By default, **ARMLET** provides some distribution functions in ``armlet.data.splitter`` and pre-defined distribution YAML config files in the ``ARMLET_DIR/configs/data/distribution`` folder.

```

## Others fields
(config_data_others)=

The `others` field includes all the other config values required to split the data.
The majority of them are the same as the parameters of [Fluke's data other fields](https://makgyver.github.io/fluke/config_data.html#other-fields):

- `sampling_perc`: sampling percentage when loading the dataset during a training iteration;

- `client_split`: percentage of the client’s data that will be used as its test set;

- `client_val_split`: percentage of the client’s test set that will be used as a validation set;

- `keep_test`: specifies whether you want to keep the test set as provided by the dataset loading function;

- `server_test`: specifies whether the server has a test set;

- `server_test_union`: specifies whether the server test set is the union of all client test sets (also apply for the server val set).
It requires to set `server_test` and `keep_test` to `false`;

- `server_split`: size of the server split with respect to the entire dataset (only used when `keep_test` is set to `false` and `server_test` is set to `true`);

- `server_val_split`: percentage of the server's data that will be used as a server validation set;

- `uniform_test`: specifies whether to use a client-side IID test set distribution regardless of the training data distribution.

```{eval-rst}

.. seealso::
	For more information about ``sampling_perc``, ``keep_test``, ``server_test``, ``client_split``, ``server_split``, and ``uniform_test``, see `Data configuration <https://makgyver.github.io/fluke/config_data.html#other-fields>`_ from Fluke documentation.

```

## Data processing configuration
(config_data_processing)=

[TODO]

## Data cleaning configuration
(config_data_cleaning)=

[TODO]

## Data loading configuration
(config_data_loading)=

[TODO]

## Data saving configuration
(config_data_saving)=

[TODO]
