r"""
Power spectrum

This module implements the ell-dependent component of common foreground
contaminants.

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""

import os
import pkg_resources
import numpy as np
from .model import Model


def _get_power_file(model):
    """ File path for the named model
    """
    data_path = './plikmflike/fgspectra/data/'#pkg_resources.resource_filename('.fgspectra', 'data/')
    filename = os.path.join(data_path, 'cl_%s.dat'%model)
    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+filename+' found in ' + str(data_path) + ' ls: ' + str(os.listdir(data_path)))


class PowerSpectrumFromFile(Model):
    """Power spectrum loaded from file(s)

    Parameters
    ----------
    filenames: array_like of strings
        File(s) to load. It can be a string or any (nested) sequence of strings

    Examples
    --------

    >>> ell = range(5)

    Power spectrum of a single file

    >>> my_file = 'cl.dat'
    >>> ps = PowerSpectrumFromFile(my_file)
    >>> ps(ell).shape
    (5)
    >>> ps = PowerSpectrumFromFile([my_file])  # List
    >>> ps(ell).shape
    (1, 5)

    Two correlated components

    >>> my_files = [['cl_comp1.dat', 'cl_comp1xcomp2.dat'],
    ...             ['cl_comp1xcomp2.dat', 'cl_comp2.dat']]
    >>> ps = PowerSpectrumFromFile(my_files)
    >>> ps(ell).shape
    (2, 2, 5)

    """

    def __init__(self, filenames, **kwargs):
        """

        The file format should be two columns, ell and the spectrum.
        """
        filenames = np.array(filenames)
        self._cl = np.empty(filenames.shape+(0,))

        for i, filename in np.ndenumerate(filenames):
            ell, spec = np.genfromtxt(filename, unpack=True)
            ell = ell.astype(int)
            # Make sure that the new spectrum fits self._cl
            n_missing_ells = ell.max() + 1 - self._cl.shape[-1]
            if n_missing_ells > 0:
                pad_width = [(0, 0)] * self._cl.ndim
                pad_width[-1] = (0, n_missing_ells)
                self._cl = np.pad(self._cl, pad_width,
                                  mode='constant', constant_values=0)

            self._cl[i+(ell,)] = spec
        self.set_defaults(**kwargs)

    def eval(self, ell=None, ell_0=None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        return amp * self._cl[..., ell] / self._cl[..., ell_0, np.newaxis]

class tSZ_Planck(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich from Planck template."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__(_get_power_file('tsz_planck'))
	
    def eval(self, ell=None, ell_0=None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        return amp * self._cl[..., ell] / self._cl[..., 3000 - 1, np.newaxis]

class kSZ_Planck(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich from Planck template."""

    def __init__(self, **kwargs):
        """
        Planck template only has Cl column (no ells)
        """
        self._cl = np.empty((0,))
        spec = np.genfromtxt(_get_power_file('ksz_planck'))
        ell = np.arange(2, spec.shape[0]+2).astype(int)
        # Make sure that the new spectrum fits self._cl
        n_missing_ells = ell.max() + 1 - self._cl.shape[-1]
        if n_missing_ells > 0:
            pad_width = [(0, 0)] * self._cl.ndim
            pad_width[-1] = (0, n_missing_ells)
            self._cl = np.pad(self._cl, pad_width,
                              mode='constant', constant_values=0)

        self._cl[ell,] = spec
        self.set_defaults(**kwargs)

    def eval(self, ell=None, ell_0=None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        return amp * self._cl[..., ell] / self._cl[..., 3000, np.newaxis]

class CIB_Planck(Model):
    """Planck CIB template.
       HTJ - after plik_v22 FORTRAN
       
      Do not use this. It is hacked together because the Planck template is
      not really designed to work well within fgspectra."""

    def __init__(self, **kwargs):
        spec = np.genfromtxt(_get_power_file('cib_planck'), unpack = False, dtype = np.float)
        ell = spec[:,0].astype(int)
        self._cl = np.zeros((max(ell)+1, 4))
        self._cl[ell,0] = spec[:,1] * (4096.68168783 / 1e6) ** 2.0
        self._cl[ell,1] = spec[:,7] * (2690.05218701 / 1e6) ** 2.0
        self._cl[ell,2] = spec[:,8] * (2690.05218701 / 1e6) * (2067.43988919 / 1e6)
        self._cl[ell,3] = spec[:,12] * (2067.43988919 / 1e6) ** 2.0
        
        ls = np.arange(self._cl.shape[0])[...,np.newaxis]
        norm = self._cl[3000, 3]
        self._cl = (self._cl / norm) * ls * (ls + 1.0) / (3000.0 * 3001.0)
        
        self.set_defaults(**kwargs)

    def eval(self, ell=None, ell_0=None, n_cib = None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        if np.isscalar(ell): ell = np.array(ell)[..., np.newaxis]
        
        return amp * self._cl[ell,:] * (ell[:,np.newaxis] / ell_0) ** (n_cib + 1.3)

class gal_Planck(Model):
    """Planck gal template.
       HTJ - after plik_v22 FORTRAN
       
      Do not use this. It is hacked together because the Planck template is
      not really designed to work well within fgspectra."""

    def __init__(self, **kwargs):
        self._cl = np.zeros((2601, 4))
        for i, filename in enumerate(['gal_planck_100', 'gal_planck_143', 'gal_planck_143x217', 'gal_planck_217']):
            ell, spec, _ = np.genfromtxt(_get_power_file(filename), unpack = True)
            
            self._cl[ell.astype(int),i] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
            self._cl[:,i] /= self._cl[200,i]
        
        self.set_defaults(**kwargs)

    def eval(self, ell=None):
        """Compute the power spectrum with the given ell and parameters."""
        if np.isscalar(ell):
            ell = np.array(ell)[..., np.newaxis]
        
        res = np.tile(np.nan, ell.shape + (4,))
        res[ell <= 2600] = self._cl[ell[ell <= 2600],:]
        return res

class tSZxCIB_Planck(PowerSpectrumFromFile):
	"""Power Spectrum for tSZxCIB from template.
	   HTJ - after ACTPolFull FORTRAN"""
	
	def __init__(self):
		super().__init__(_get_power_file('sz_x_cib_planck'))

class PowerLaw(Model):
    r""" Power law

    .. math:: C_\ell = (\ell / \ell_0)^\alpha
    """
    def eval(self, ell=None, alpha=None, ell_0=None, amp=1.0):
        """

        Parameters
        ----------
        ell: float or array
            Multipole
        alpha: float or array
            Spectral index.
        ell_0: float
            Reference ell
        amp: float or array
            Amplitude, shape must be compatible with `alpha`.

        Returns
        -------
        cl: ndarray
            The last dimension is ell.
            The leading dimensions are the hypothetic dimensions of `alpha`
        """
        alpha = np.array(alpha)[..., np.newaxis]
        amp = np.array(amp)[..., np.newaxis]
        return amp * (ell / ell_0)**alpha


class SquarePowerLaw(Model):
	r""" Square Power Law
	
	Power Law for alpha = 2 but with the low-ell term.
	
	.. math:: C_\ell = ( \ell (\ell + 1) ) / ( \ell_0 (\ell_0 + 1) )
	"""
	def eval(self, ell = None, ell_0 = None, amp = 1.0):
		"""
		Parameters
		----------
		ell: float or array
			Multipole
		ell_0: float
			Reference ell
		amp: float or array
			Amplitude, shape must be compatible with `ell`.
		
		Returns
		-------
		cl: ndarray
			Has same shape as ell.
		"""
		amp = np.array(amp)[..., np.newaxis]
		return amp * (ell * (ell + 1.0)) / (ell_0 * (ell_0 + 1.0))
