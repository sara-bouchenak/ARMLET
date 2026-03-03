(generate_doc)=

# How to generate **Armlet** documentation

**Armlet** is based on [Sphinx](https://www.sphinx-doc.org/en/master/#) for managing the documentation.

First, go to the docs folder:

```bash
cd ARMLET_DIR
cd docs
```

Then, activate the environment and install the needed dependencies:

```bash
activate armlet
pip install -r requirements.txt
```

Finally, run the following command:

```bash
make html
```
