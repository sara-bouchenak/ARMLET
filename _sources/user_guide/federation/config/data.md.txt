(config_data)=

# Data configuration

In this page, we detail the configuration values related to the data pipeline.
The configuration groups or values in `data` are:

- `seed`: the seed for the random number generator in the data pipeline, which is essential for the experiment **reproducibility**;

- [`dataset`](config_data_dataset): for loading the dataset and transforming it to the appropriate format;

- [`splitter`](config_data_splitter): for splitting the data among the clients and the server (including the data distribution when partitionning);

- [`processing`](config_data_processing): for performing data processing;

- [`cleaning`](config_data_cleaning): **optional**, for performing data cleaning;

- [`loading`](config_data_loading): **optional**, for loading static data (splitted, cleaned, tensors) that were save during previous experiments instead of loading the raw dataset;

- [`saving`](config_data_saving): **optional**, for saving data at one or multiple specific steps of the data pipeline;

- [`others`](config_data_others): for the others config values that do not fall into the other categories.

```{eval-rst}

.. seealso::
	See the tutorial `Understanding the data pipeline in armlet <https://sara-bouchenak.github.io/ARMLET/getting_started/tutorials/data_pipeline.html#>`_ to learn more about how this framework uses these configurations to perform the data pipeline.

```

## Dataset configuration
(config_data_dataset)=

The `dataset` field requires two mandatory config values:

- `dataset_name`: the name of the dataset, used by the logger;

- `_target_`: the function used to load the dataset;

- and additional config values depending on the chosen loading function (that will be dynamically pass to the function as arguments).

```{eval-rst}

.. tip::
  By default, **ARMLET** provides multiple dataset loading functions in ``armlet.data.datasets`` and pre-defined dataset YAML config files in the ``ARMLET_DIR/configs/data/dataset`` folder.

```

## Data splitter configuration
(config_data_splitter)=

The `splitter` field includes all the config values required to split the data.
The majority of them are the same as the parameters of [Fluke's data other fields](https://makgyver.github.io/fluke/config_data.html#other-fields) and [Fluke's data distribution field](https://makgyver.github.io/fluke/config_data.html#data-distribution):

- `distribution`: config group for specifying the data distribution when partitionning (similar to the one provided by Fluke, but adapted to the data format required by **ARMLET**).
The only mandatory config value in this group is `_target_`, which is the distribution function used during the data splitting step;

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

.. tip::
  By default, **ARMLET** provides some distribution functions in ``armlet.data.splitter`` and pre-defined distribution YAML config files in the ``ARMLET_DIR/configs/data/splitter/distribution`` folder.

```

```{eval-rst}

.. seealso::
	For more information about ``keep_test``, ``server_test``, ``client_split``, ``server_split``, and ``uniform_test``, see `Data configuration <https://makgyver.github.io/fluke/config_data.html#other-fields>`_ from Fluke documentation.

```

## Data processing configuration
(config_data_processing)=

The `processing` field consists of several config groups, each of which defines a processing step to be applied to the data.
Each of these config groups requires the following config values:

- `_target_`: the function corresponding to the data processing to perform;

- `_apply_directly_to_subdata_`: boolean for determining whether the processing step is applied directly to the subdata (local training data for each client, server test data, etc.). This can be usefull when the processing must access all data before processing the subdata, such as for feature encoding or normalization;

- and additional config values depending on the chosen data processing function (that will be dynamically pass to the function as arguments).

Note that the name of the config groups only serves to organize the configs and have no impact on the processing steps.

```{eval-rst}

.. important::
	The **order** of the config groups in the ``processing`` field is important as **ARMLET** will perform the different steps in this order.

```

```{eval-rst}

.. tip::
	ARMLET provides multiple data processing functions in ``armlet.data.processing`` and pre-defined YAML config files in the ``ARMLET_DIR/configs/data/processing`` folder.

```

## Data cleaning configuration
(config_data_cleaning)=

The `cleaning` field is optional and consists of several config groups, each of which defines a cleaning step to be applied to the data.
Each of these config groups (the name being the type of corrupted data/error to be handled) requires the following config values:

- `_target_`: the class corresponding to the data cleaning step to perform;

- and additional config values depending on the chosen data cleaning function (that will be dynamically pass to the function as arguments).

For the moment, only `armlet.data.cleaning.missing_values.RemoveMV` is implemented in **ARMLET**.
It removes any sample containing missing values (i.e., NaN).
Feel free to implement your own methods for handling your type of corrupted data/error (e.g., missing values, label errors, outliers).

Note that the `cleaning` field has also a config value `name` that is only used to give a name to the combination of cleaning steps (for logging).

## Data loading configuration
(config_data_loading)=

The `loading` field can be used to load static data (splitted, cleaned, tensors) that were save during previous experiments (using the saving module).
The config values are:

- `static`: boolean for activating the static data loading;

- `load_dir`: directory from which the data will be loaded.

## Data saving configuration
(config_data_saving)=

The `saving` field can be used to save the data at one or multiple specific steps of the data pipeline.
The config values are:

- `save_dir`: directory where the data will be saved. A subfolder will be created, with a name that depends on the type of saving (e.g., "after_processing", "no_cleaning", etc.).

- `save_data_before_cleaning`: boolean for saving data before cleaning (set to `False` by default);

- `save_data_after_cleaning`: boolean for saving data after cleaning (set to `False` by default);

- `save_data_after_processing`: boolean for saving data after processing (set to `False` by default). This can be particularly usefull when dealing with images as it will save tensors instead of raw images.

## Others fields
(config_data_others)=

The `others` field includes all the others parameters that do not fall into the other categories.

- `sampling_perc`: sampling percentage when loading the dataset during a training iteration.

```{eval-rst}

.. seealso::
	For more information about ``sampling_perc``, see `Data configuration <https://makgyver.github.io/fluke/config_data.html#other-fields>`_ from Fluke documentation.

```
