# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants.

This package draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).

	NOTE.
	THIS IS A MINIMAL COPY OF FGSPECTRA. All unused spectra, templates and other
	implemented functions are removed. This copy is designed to be used as-is
	with MFLikePlik as implemented by Hidde Jense. If you seek to use fgspectra
	for a different project, please see https://github.com/simonsobs/fgspectra.
"""

import numpy as np
from scipy import constants
from .model import Model
from functools import wraps


T_CMB = 2.72548
H_OVER_KT_CMB = constants.h * 1e9 / constants.k / T_CMB

def _rj2cmb(nu):
    x = H_OVER_KT_CMB * nu
    return (np.expm1(x) / x)**2 / np.exp(x)

class PowerLaw(Model):
    r""" Power Law

    .. math:: f(\nu) = (\nu / \nu_0)^{\beta}
    """
    def eval(self, nu=None, beta=None, nu_0=None):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in the same units as `nu_0`. If array, the shape is
            ``(freq)``.
        beta: float or array
            Spectral index. If array, the shape is ``(...)``.
        nu_0: float or array
            Reference frequency in the same units as `nu`. If array, the shape
            is ``(...)``.

        Returns
        -------
        sed: ndarray
            If `nu` is an array, the shape is ``(..., freq)``.
            If `nu` is scalar, the shape is ``(..., 1)``.
            Note that the last dimension is guaranteed to be the frequency.

        Note
        ----
        The extra dimensions ``...`` in the output are the broadcast of the
        ``...`` in the input (which are required to be broadcast-compatible).

        Examples
        --------

        - T, E and B synchrotron SEDs with the same reference frequency but
          different spectral indices. `beta` is an array with shape ``(3)``,
          `nu_0` is a scalar.

        - SEDs of synchrotron and dust (approximated as power law). Both `beta`
          and `nu_0` are arrays with shape ``(2)``

        """
        beta = np.array(beta)[..., np.newaxis]
        if np.isscalar(nu_0): nu_0 = np.array(nu_0)[..., np.newaxis]
        return (nu / nu_0)**beta * (_rj2cmb(nu) / _rj2cmb(nu_0))

class ConstantSED(Model):
    """Frequency-independent component."""

    def eval(self, nu=None, amp=1.):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            It just determines the shape of the output.
        amp: float or array
            Amplitude (or set of amplitudes) of the constant SED.

        Returns
        -------
        sed: ndarray
            If `nu` is an array, the shape is ``amp.shape + (freq)``.
            If `nu` is scalar, the shape is ``amp.shape + (1)``.
            Note that the last dimension is guaranteed to be the frequency.
        """
        amp = np.array(amp)[..., np.newaxis]
        return amp * np.ones_like(np.array(nu))
