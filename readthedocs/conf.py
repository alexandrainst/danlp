# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from recommonmark.transform import AutoStructify

# -- Project information -----------------------------------------------------

project = 'DaNLP'
copyright = '2020, Alexandra Institute'
author = 'Alexandra Institute'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'recommonmark',
        'sphinx.ext.autodoc',
        'sphinx_markdown_tables',
        'sphinx.ext.todo',
        'sphinx.ext.autosectionlabel'
]

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
        'logo_only': True,
        'display_version': True,
        'prev_next_buttons_location': 'bottom',
        'style_external_links': True,
        # Toc options
        'collapse_navigation': False,
        'sticky_navigation': False,
        'navigation_depth': 3,
        'includehidden': False,
        'titles_only': False
}

html_title = "DaNLP documentation"
html_logo = "docs/imgs/danlp_logo.png"
html_favicon = "docs/imgs/danlp_logo.png"
#html_style = 'custom.css'

master_doc = 'index'

github_doc_root = 'https://github.com/alexandrainst/danlp/tree/master/readthedocs/docs'


autosectionlabel_prefix_document = True

def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)