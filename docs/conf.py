import re
from os import path as op

with open(op.join(op.dirname(op.dirname(__file__)), "pyproject.toml"), "r") as poetry_file:
    for line in poetry_file.readlines():
        try:
            version = re.search(r"version\s*\t*=\s*\t*['\"](.*?)['\"]", line).groups()[0]
            break
        except AttributeError:
            pass


# -- Project information -----------------------------------------------------

project = 'crypto_history'
copyright = '2021, Vikramaditya Gaonkar'
author = 'Vikramaditya Gaonkar'

# The full version, including alpha/beta/rc tags
release = str(version)


def no_namedtuple_attrib_docstring(app, what, name,
                                   obj, options, lines):
    is_namedtuple_docstring = (
        len(lines) == 1 and
        'Alias for field number' in lines[0]
    )
    if is_namedtuple_docstring:
        # We don't return, so we need to purge in-place
        print(lines)
        del lines[:]


def setup(app):
    app.connect(
        'autodoc-process-docstring',
        no_namedtuple_attrib_docstring,
    )


# -- General configuration ---------------------------------------------------
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    'sphinx.ext.napoleon'
]
intersphinx_mapping = {'python': ('https://docs.python.org/3.8', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']