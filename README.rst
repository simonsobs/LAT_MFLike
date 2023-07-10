=============================
LAT Multifrequency Likelihood
=============================

An external likelihood using `cobaya <https://github.com/CobayaSampler/cobaya>`_.

.. image:: https://img.shields.io/pypi/v/mflike.svg?style=flat
   :target: https://pypi.python.org/pypi/mflike

.. image:: https://img.shields.io/github/actions/workflow/status/simonsobs/LAT_MFLike/testing.yml?branch=master
   :target: https://github.com/simonsobs/LAT_MFLike/actions

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/simonsobs/LAT_MFLike/master?filepath=notebooks%2Fmflike_tutorial.ipynb


.. image:: https://codecov.io/gh/simonsobs/LAT_MFLike/branch/master/graph/badge.svg?token=qrrVcbNCs5
   :target: https://codecov.io/gh/simonsobs/LAT_MFLike


Installing the code
-------------------

The easiest way to install and to use ``mflike`` likelihood is *via* ``pip``

.. code:: shell

    pip install mflike

If you want to dig into the code, you'd better clone this repository to some location

.. code:: shell

    git clone https://github.com/simonsobs/LAT_MFLike.git /where/to/clone

Then you can install the ``mflike`` likelihood and its dependencies *via*

.. code:: shell

    pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``mflike`` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Installing LAT likelihood data
------------------------------

Preliminary simulated data can be found at `NERSC
<https://portal.nersc.gov/cfs/sobs/users/MFLike_data>`_. You can download them by yourself but you
can also use the ``cobaya-install`` binary and let it do the installation job. For instance, if you
do the next command

.. code:: shell

    cobaya-install /where/to/clone/examples/mflike_example.yaml -p /where/to/put/packages

data and code such as `CAMB <https://github.com/cmbant/CAMB>`_ will be downloaded and installed
within the ``/where/to/put/packages`` directory. For more details, you can have a look to ``cobaya``
`documentation <https://cobaya.readthedocs.io/en/latest/installation_cosmo.html>`_.

Running/testing the code
------------------------

You can test the ``mflike`` likelihood by doing

.. code:: shell

    cobaya-run /where/to/clone/examples/mflike_example.yaml -p /where/to/put/packages

which should run a MCMC sampler for the simulation n°44 (*i.e.* ``data_sacc_00044.fits`` in the
``mflike_example.yaml`` file) using the combination of TT, TE and EE spectra (*i.e.*
``polarizations: ['TT', 'TE', 'ET', 'EE']``). The results will be stored in the ``chains/mcmc``
directory.

Simulation releases
-------------------

- ``v0.8`` release has spectra produced with high accuracy camb parameters up to ``ell = 9000``, the foreground components are integrated in frequency using top-hat bandpasses, stored in the ``sacc`` files

- ``v0.7.1`` is a bugfix release of ``v0.7`` release where the mono-frequency and dirac bandpasses
  have been correctly set with the ``sacc`` files

- ``v0.7`` release includes the ACT like foregrounds. Simulation parameters are also stored within ``sacc`` metadata and the associated ``dict`` file can be viewed `here <https://gist.github.com/xgarrido/5d2fdbe4232cfa9ad1156ee30baa7811>`_.

- ``v0.6`` release make use of CMB map based simulations (see https://github.com/simonsobs/map_based_simulations/blob/master/201911_lensed_cmb/README.md). Only temperature foregrounds were considered.

Releases prior to ``v0.6`` version were development simulations and should not be used for simulation analysis.
