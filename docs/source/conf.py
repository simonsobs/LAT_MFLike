from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("mflike").version
except DistributionNotFound:
    __version__ = "unknown version"

import os
import sys

sys.path.append(os.path.abspath("../.."))
# autodoc_mock_imports = ["numpy", "cobaya"]

# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = ".rst"
master_doc = "index"

project = "mflike"
copyright = "2019 - 2024, Simons Observatory Collaboration Analysis Library Task Force"
author = "Simons Observatory Collaboration Power Spectrum Group"
language = "en"
version = __version__
release = __version__

exclude_patterns = ["_build"]


# HTML theme
html_theme = "sphinx_book_theme"
# Add paths to extra static html files from notebook conversion
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "Multifrequency Likelihood for SO Large Aperture Telescope"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "logo": {
        "alt_text": "LAT_MFLike - Home",
        "text": html_title,
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "path_to_docs": "docs",
    "repository_url": "https://github.com/simonsobs/LAT_MFLike",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "off"
nb_execution_timeout = -1
