=============================
LAT Multifrequency Likelihood
=============================


External likelihood using `cobaya <https://github.com/CobayaSampler/cobaya>`_.

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
remove the ``--develop`` option.

Testing the code
----------------

You can test the ``mflike`` likelihood by doing

.. code:: shell

    $ cobaya-run /where/to/clone/tests/test_mflike.yaml

which should run a MCMC sampler. Make sure `CAMB <https://github.com/cmbant/CAMB>`_ is properly installed (for instance, by typing ``pip install camb``).
