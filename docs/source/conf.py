# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = 'ARMLET'
copyright = '2026, Baudouin NALINE'
author = 'Baudouin NALINE'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx_inline_tabs',
    'sphinx.ext.apidoc',
    "sphinx.ext.githubpages",
]

myst_enable_extensions = ["colon_fence"]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {
        "text": "ARMLET",
    }
}

apidoc_modules = [
    {
        'path': '../../armlet',
        'destination': 'api',
        'exclude_patterns': ['**/FL_decentralized/*', '**/results_analysis/*'],
        'max_depth': 4,
        'follow_links': False,
        'separate_modules': True,
        'include_private': False,
        'no_headings': False,
        'module_first': False,
        'implicit_namespaces': False,
        'automodule_options': {
            'members', 'show-inheritance', 'undoc-members'
        },
    },
]
