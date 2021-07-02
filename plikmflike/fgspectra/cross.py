r"""
Models of cross-spectra

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).

	NOTE.
	THIS IS A MINIMAL COPY OF FGSPECTRA. All unused spectra, templates and other
	implemented functions are removed. This copy is designed to be used as-is
	with MFLikePlik as implemented by Hidde Jense. If you seek to use fgspectra
	for a different project, please see https://github.com/simonsobs/fgspectra.
"""
from abc import ABC, abstractmethod
import numpy as np
import pprint
from scipy import constants
from . import frequency as fgf
from . import power as fgp
from .frequency import T_CMB, _rj2cmb # hack
from .model import Model

class FactorizedCrossSpectrum(Model):
    r"""Factorized cross-spectrum

    Cross-spectrum of **one** component for which the scaling in frequency
    and in multipoles are factorizable

    .. math:: xC_{\ell}^{(ij)} = f(\nu_j) f(\nu_i) C_{\ell}

    Parameters
    ----------
    sed : callable
        :math:`f(\nu)`. It returns an array with shape ``(..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl_args : callable
        :math:`C_\ell`. It returns an array with shape ``(..., ell)``.
        It can be :class:`fgspectra.power.PowerSpectrum`

    Note
    ----
    The two (optional) sets of extra dimensions ``...`` must be
    broadcast-compatible.
    """

    def __init__(self, sed, cl, **kwargs):
        self._sed = sed
        self._cl = cl
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'sed_kwargs' in kwargs:
            self._sed.set_defaults(**kwargs['sed_kwargs'])
        if 'cl_kwargs' in kwargs:
            self._cl.set_defaults(**kwargs['cl_kwargs'])

    @property
    def defaults(self):
        return {
            'sed_kwargs': self._sed.defaults,
            'cl_kwargs': self._cl.defaults
        }

    def _get_repr(self):
        sed_repr = self._sed._get_repr()
        key = list(sed_repr.keys())[0]
        sed_repr[key + ' (SED)'] = sed_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + ' (Cl)'] = cl_repr.pop(key)

        return {type(self).__name__: [sed_repr, cl_repr]}

    def eval(self, sed_kwargs={}, cl_kwargs={}):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_args : list
            Arguments for which the `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """
        f_nu = self._sed(**sed_kwargs)[..., np.newaxis]
        return f_nu[..., np.newaxis] * f_nu * self._cl(**cl_kwargs)

class PlankCrossSpectrum(Model):
    r"""Planck multi-template Cross Spectrum.
    	I do not recommend using this elsewhere, it is designed to be single-purpose.
    	If a similar template is needed, I recommend writing a more stable, multi-purpose implementation of this template.
    	
    	HTJ.
    """

    def __init__(self, sed, cl, **kwargs):
        self._sed = sed
        self._cl = cl
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'sed_kwargs' in kwargs:
            self._sed.set_defaults(**kwargs['sed_kwargs'])
        if 'cl_kwargs' in kwargs:
            self._cl.set_defaults(**kwargs['cl_kwargs'])

    @property
    def defaults(self):
        return {
            'sed_kwargs': self._sed.defaults,
            'cl_kwargs': self._cl.defaults
        }

    def _get_repr(self):
        sed_repr = self._sed._get_repr()
        key = list(sed_repr.keys())[0]
        sed_repr[key + ' (SED)'] = sed_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + ' (Cl)'] = cl_repr.pop(key)

        return {type(self).__name__: [sed_repr, cl_repr]}

    def eval(self, sed_kwargs={}, cl_kwargs={}):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_args : list
            Arguments for which the `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """
        f_nu = self._sed(**sed_kwargs)[..., np.newaxis]
        c_ell = self._cl(**cl_kwargs)
        
        a = f_nu[...,np.newaxis] * f_nu
        
        x = np.zeros((a.shape[0], a.shape[1], c_ell.shape[0], c_ell.shape[1]))
        for i in range(x.shape[-1]):
            A = a[:,:,0]
            B = c_ell[:,i]
            x[...,i] = A[...,np.newaxis] * B
        
        return x
