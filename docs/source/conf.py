# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
# print(sys.path)

try:
    import sphinx_bootstrap_theme
except ModuleNotFoundError:
    from pip._internal import main as pipmain
    failed = pipmain(["install", 'sphinx_bootstrap_theme'])
    import sphinx_bootstrap_theme
except ImportError:
    from pip._internal import main as pipmain
    failed = pipmain(["install", 'sphinx_bootstrap_theme'])
    import sphinx_bootstrap_theme

import recommonmark
from recommonmark.transform import AutoStructify

autodoc_default_flags = ['members']
autosummary_generate = True

# -- Project information -----------------------------------------------------
project = 'eonr'
copyright = '2019, Tyler J Nigon'
author = 'Tyler J Nigon'

# The short X.Y version
version = '1.0.0'
# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Creates TOC sub-level for EONR methods
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'recommonmark',
    'nbsphinx',  # converts Jupyter notebooks to html
    'sphinx.ext.napoleon'  # for parsing docstrings
]

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

language = 'python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Keeps sphinx from reading files in this folder. This does not affect
# sphinx-apidoc (must add it again when creating docs)
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
# html_theme = 'sphinx_rtd_theme'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "EONR",
    'navbar_links': [
        ('Github', "https://github.com/tnigon/eonr", True)
    ],
    'navbar_site_name': "Contents",
    'globaltoc_depth': 1,
    'navbar_pagenav': False,  # sidebar is doing this
    # 'navbar_pagenav_name': "Page Menu",
    'navbar_fixed_top': "true",
    'bootswatch_theme': "Flatly",
    'bootstrap_version': "3",
    }

html_sidebars = {
   '**': ['localtoc.html', 'relations.html', 'searchbox.html'],
   'using/windows': ['windowssidebar.html', 'searchbox.html'],
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'eonrdoc'


# -- Options for LaTeX output ------------------------------------------------
mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
}

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'eonr.tex', 'EONR Documentation',
     'Tyler J Nigon', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'EONR', 'EONR Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'EONR', 'EONR Documentation',
     author, 'EONR', 'Calculates the economic optimum nitrogen rate and plots the results',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/doc/'
def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)
