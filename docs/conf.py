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
import shutil


# - Copy over examples folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images
# from gpytorch docs source

examples_source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)
            shutil.copyfile(source_filename, dest_filename)

# -- Project information -----------------------------------------------------

project = "LANTERN"
copyright = "2021, Dr. Peter D. Tonner"
author = "Dr. Peter D. Tonner"

# The full version, including alpha/beta/rc tags
release = "0.2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

# maybe get better docs for attrs?: https://github.com/agronholm/sphinx-autodoc-typehints/issues/44
autodoc_typehints = "description"  # show type hints in doc body instead of signature
autoclass_content = "both"  # get docstring from class level and init simultaneously

intersphinx_mapping = {
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "gpytorch": ("https://docs.gpytorch.ai/en/latest/", None),
    "python": ("http://docs.python.org/3", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
