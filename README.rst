===========
MFLike-plik
===========

An ``MFLike`` implementation of the Planck ``plik`` high-ell likelihood for `cobaya <https://github.com/CobayaSampler/cobaya>`_.

Running/testing the code
------------------------

You can test the ``MFLike-plik`` likelihood by including the ``PlikMFLike`` in your ``likelihoods`` block. All parameters used by ``plik`` have the same name in ``MFLike-plik``.

You need to unzip the ``data/covmat.zip`` file to include the ``data/covmat.dat`` file. After that you can test the code by running

.. code:: shell

    $ python3 -m cobaya run mflike-plik.yaml

Included files
--------------

The following files and folders are included:

    ``plikmflike/`` contains the main ``PlikMFLike`` code base.

    ``plikmflike/fgspectra/`` contains a custom ``fgspectra`` implementation that is needed for the correct ``plik`` foreground modeling. It contains the Planck foreground models in the ``fgspectra/data/`` folder. Foreground model implementations are included that replicate the Planck foregrounds based on the original templates.

    ``data/`` is the default path to place the ``plik`` data files (because of ``github`` filesize limits, they are not included here, 

Two example ``.yaml`` files are incldued:

    ``mflike-plik.yaml`` runs a basic mcmc chain over the data.
    
    ``mflike-plik-lowLE.yaml`` couples the ``MFLike-Plik`` code to the ``plik-lowl`` and ``plik-lowE`` codes and performs a combined chain over them. It should be modified to point to the local ``.clik`` files (see the `cobaya documentation <https://cobaya.readthedocs.io/en/latest/likelihood_planck.html>`_ for more information on that).
    
    The remaining ``params_*.yaml`` files define the parameters and priors as originally presented in the `cobaya plik highl <https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods/planck_2018_highl_plik>`_ implementation.
    
The sample chains should be able to be ran as

.. code:: shell

    $ python3 -m cobaya run mflike-plik.yaml

It should be noted that the code should be run using ``python 3``, it is incompatible with ``python 2``. For the ``mflike-plik-lowLE.yaml`` example, you should direct it to your local ``.clik`` files as mentioned above.
