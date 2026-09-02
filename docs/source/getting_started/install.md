(install)=

# Installation

First, clone the **ARMLET** repository:

```bash
git clone https://github.com/sara-bouchenak/ARMLET.git
```

## Setup the environment

1. Install [conda](https://www.anaconda.com/docs/getting-started/main) for managing the environments.

2. Run the following commands:

```bash
conda create -n armlet python=3.13.5
conda activate armlet
```

## Install **ARMLET**

Install **``armlet``** using `pip`:

```bash
cd ARMLET_DIR
pip install .
```

## Initialize your project

Initialize a new project by creating an empty folder ``PROJECT_DIR``.

In ``PROJECT_DIR``, create the folder ``configs_armlet`` so you can add your own configuration files later: 

```bash
cd PROJECT_DIR
mkdir configs_armlet
```

```{eval-rst}

.. important::
	In the documentation, we use ``ARMLET_DIR`` to refer to the path of the **ARMLET** project and ``PROJECT_DIR`` to the path of your current project in which you use **ARMLET** and will run the command ``armlet``.

```

Then, create the Python file ``link_configs_folder.py`` with the following content, and place it in the ``PROJECT_DIR/hydra_plugins`` directory.
It will permits to automatically link the new config files of your project (located in ``PROJECT_DIR/configs_armlet``) to ``armlet``.

```python
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class CustomSearchPathPlugin(SearchPathPlugin):

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="project", path="file://configs_armlet"
        )
```

Now everything is ready to [run your first experiment](gs_run_first_exp) with **ARMLET**.

## Software requirements

**``armlet``** were developed with Python 3.13.5.

All dependencies listed in `ARMLET_DIR/requirements.txt` are needed (they are installed in the same time when installing **``armlet``** with `pip`).
