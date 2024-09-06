r"""
Simple code to generate the bandpass shift dictionary needed in the presence of bandpass shifts different from 0.
We compute simple gaussian beams for :math:`\nu_0 + \Delta \nu`, assuming a diffraction limited experiment.
This is thought to be used in the absence of data coming from the planets beams measurements.
"""

import numpy as np
from astropy import constants, units
import os
import tempfile
import yaml
from cobaya.install import install

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

data_path = packages_path + "/data/MFLike/v0.8"

install({"likelihood": {"mflike.TTTEEE": None}}, path=packages_path)

def compute_FWHM(nu):
    """
    Simple function to compute FWHM for the LAT assuming a diffraction limited experiment.

    :param nu: the frequency array in GHz

    :return: the FWHM for each nu
    """
    mirror_size = 6 * units.m
    wavelenght = constants.c / (nu * 1e9 / units.s)
    fwhm = 1.22 * wavelenght / mirror_size
    return fwhm

def gauss_beams(fwhm0, nu, nu0, alpha, lmax, pol):
    r"""
    Computes the Gaussian beam (either for T or pol) for each frequency of a
    frequency array according to eqs. 54/55 of arXiv:astro-ph/0008228. We assume a more general
    scaling for the FWHM: :math:`FWHM(\nu) = FWHM(\nu_0) \left( \frac{\nu}{\nu_0} \right)^{-\alpha}`.

    :param fwhm0: the FWHM for the pivot frequency
    :param nu: the frequency array in GHz
    :param nu0: the pivot frequency in GHz
    :param alpha: the exponent of the frequency scaling
                  :math:`\left( \frac{\nu}{\nu_0} \right)^{-\alpha/2}`
    :param lmax: the lmax of the beams
    :param pol: (Bool) False to compute temperature Gaussian beam,
                True for the polarization one

    :return: a :math:`b^{Gauss.}_{\ell}(\nu)` = ``array(freqs, lmax +2)`` with Gaussian beam
             profiles for each frequency in :math:`\nu` (from :math:`\ell = 0`)
    """
    from astropy import constants, units
    import healpy as hp

    fwhm = fwhm0 * (nu / nu0)**(-alpha/2.)
    bls = np.empty((len(nu), lmax + 1))
    for ifw, fw in enumerate(fwhm):
        # saving the beam from ell = 2 to ell max of l_bpws
        if not pol:
            bls[ifw, :] = hp.gauss_beam(fw, lmax=lmax)
        else:
            # selecting the polarized gaussian beam
            bls[ifw, :] = hp.gauss_beam(fw, lmax=lmax, pol=True)[:, 1]

    return bls

beam_dnu_dict = {}
# express delta nu as floats for the correct behavior of the code that will read this dictionary
dnu = np.arange(-20, 21, dtype = float)
for f in [93, 145, 225]:
    fwhm = compute_FWHM(f)
    gbeamT = gauss_beams(fwhm, f+dnu, f, 2, 10000, False)
    gbeamP = gauss_beams(fwhm, f+dnu, f, 2, 10000, True)
    beam_dnu_dict[f"LAT_{f}_s0"] = {
            "beams": {f"{dn}": gbeamT[idn] for idn, dn in enumerate(dnu)},
            "nu_0": f,
            "alpha": 2
            }
    beam_dnu_dict[f"LAT_{f}_s2"] = {
            "beams": {f"{dn}": gbeamP[idn] for idn, dn in enumerate(dnu)},
            "nu_0": f,
            "alpha": 2
            }


# saving the yaml file
with open(data_path + '/LAT_beam_bandshift.yaml', 'w') as file:
    yaml.dump(beam_dnu_dict, file, default_flow_style=False)
    print("saving "+data_path + '/LAT_beam_bandshift.yaml')


