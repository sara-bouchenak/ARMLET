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

```{eval-rst}

.. important::
	In the documentation, we use ``ARMLET_DIR`` to refer to the path of the **ARMLET** project and ``PROJECT_DIR`` to the path of your current project in which you use **ARMLET** and run the command ``armlet``.

```

Now everything is ready to [run your first experiment](run_first_exp) with **ARMLET**.

## Software requirements

**``armlet``** were developed with Python 3.13.5.

All dependencies listed in `ARMLET_DIR/requirements.txt` are needed (they are installed in the same time when installing **``armlet``** with `pip`).
