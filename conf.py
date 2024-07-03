# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


project = 'hyphi gym'
copyright = '2023, Philipp Altmann'
author = 'Philipp Altmann'
exclude_patterns = ['docs', 'Thumbs.db', '.DS_Store', 'TODO.md', '.direnv']
html_theme = 'alabaster'

extensions = ['autoapi.extension', 'myst_parser']
source_suffix = [".rst"]
html_theme = "furo"

autoapi_type = 'python'
autoapi_dirs = ['hyphi_gym' ]
autoapi_add_toctree_entry = False

# https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_enable_extensions = ['dollarmath']
myst_links_external_new_tab = True
