=============================
LAT Multifrequency Likelihood
=============================

An external likelihood using `cobaya <https://github.com/CobayaSampler/cobaya>`_.

.. image:: https://travis-ci.com/simonsobs/LAT_MFLike.svg?branch=master
   :target: https://travis-ci.com/simonsobs/LAT_MFLike

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/simonsobs/LAT_MFLike/master?filepath=notebooks%2Fmflike_tutorial.ipynb


Installing the code
-------------------

You first need to clone this repository to some location

.. code:: shell

    $ git clone https://github.com/simonsobs/LAT_MFLike.git /where/to/clone

Then you can install the ``mflike`` likelihood and its dependencies *via*

.. code:: shell

    $ pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``mflike`` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

**NB:** If ``cobaya`` is already installed on your system, make sure to uninstall it first or to run
the previous ``pip install`` command in a virtual python environment where no ``cobaya``
installation has been done.

Installing LAT likelihood data
------------------------------

Preliminary simulated data can be found in the ``LAT_MFLike_data`` `repository
<https://github.com/simonsobs/LAT_MFLike_data>`_. You can download them by yourself but you can also
use the ``cobaya-install`` binary and let it do the installation job. For instance, if you do the
next command

.. code:: shell

    $ cobaya-install /where/to/clone/tests/test_mflike.yaml -p /where/to/put/packages

data and code such as `CAMB <https://github.com/cmbant/CAMB>`_ will be downloaded and installed
within the ``/where/to/put/packages`` directory. For more details, you can have a look to ``cobaya``
`documentation <https://cobaya.readthedocs.io/en/latest/installation_cosmo.html>`_.

Running/testing the code
------------------------

You can test the ``mflike`` likelihood by doing

.. code:: shell

    $ cobaya-run /where/to/clone/tests/test_mflike.yaml -p /where/to/put/packages

which should run a MCMC sampler for the first simulation (*i.e.* ``sim_id: 0`` in the
``test_mflike.yaml`` file) using the combination of TT, TE and EE spectra (*i.e.* ``select:
tt-te-ee``). The results will be stored in the ``chains/mcmc`` directory.
