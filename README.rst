===========
MFLike-plik
===========

An ``MFLike`` implementation of the Planck ``plik`` high-ell likelihood for `cobaya <https://github.com/CobayaSampler/cobaya>`_.

Running/testing the code
------------------------

You can test the ``MFLike-plik`` likelihood by including the ``PlikMFLike`` in your ``likelihoods`` block. All parameters used by ``plik`` have the same name in ``MFLike-plik``.

Included files
--------------

The following files and folders are included:
    `data/` contains the original `plik` data files for the bins, spectra, covariance matrix and systematics.

    `examples/` contains several sample `.yaml` files for a variety of chains.
        `lcdm.yaml` is an example that runs a basic LCDM chain using only `MFLike-plik`.

        `mf-TTTEEE.yaml` is an example that couples `MFLike-plik` to a (locally-installed) `plik-lowL` and `plik-lowE` likelihoods.
