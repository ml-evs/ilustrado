#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ilustrado documentation build configuration file, created by
# sphinx-quickstart on Tue Jan  3 17:43:56 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinxarg.ext',
              'sphinx.ext.mathjax',
              'sphinxcontrib.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autodoc_member_order = 'bysource'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'ilustrado'
copyright = '2017-2018, Matthew Evans'
author = 'Matthew Evans'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.3b'
# The full version, including alpha/beta/rc tags.
release = '0.3b'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests', '*tests*', 'setup.py']
exclude_path = ['ilustrado/tests']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# run apidoc automatically on RTD: https://github.com/rtfd/readthedocs.org/issues/1139
def run_apidoc(_):
    import subprocess
    import glob
    src_dir = os.path.abspath(os.path.dirname(__file__))
    excludes = []
    # excludes = glob.glob(os.path.join(src_dir, '../../ilustrado/tests/'))
    module = os.path.join(src_dir, '../../ilustrado')
    cmd_path = 'sphinx-apidoc'
    print(excludes)
    command = [cmd_path, '-M', '-o', src_dir, module, ' '.join(excludes)]
    print(command)
    subprocess.check_call(command)

def setup(app):
    app.connect('builder-inited', run_apidoc)


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_them_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ilustradodoc'


# -- Options for LaTeX output ---------------------------------------------

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
    (master_doc, 'ilustrado.tex', 'ilustrado Documentation',
     'Matthew Evans', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ilustrado', 'ilustrado Documentation',
     [author], 1)
]

intersphinx_mapping = intersphinx_mapping = {'python': ('https://docs.python.org/3.6', None),
                                             'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                                             'pymongo': ('https://api.mongodb.com/python/current/', None),
                                             'matador': ('https://matador-db.readthedocs.io/en/latest/', None),
                                             'np': ('http://docs.scipy.org/doc/numpy/', None),
                                             'matplotlib': ('http://matplotlib.org', None)}


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ilustrado', 'ilustrado Documentation',
     author, 'ilustrado', 'One line description of project.',
     'Miscellaneous'),
]
