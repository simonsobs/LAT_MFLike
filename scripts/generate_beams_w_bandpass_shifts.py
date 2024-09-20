r"""
Simple code to generate a yaml file with Gaussian beams and the beam dictionary needed in the presence of bandpass shifts different from 0. Both are needed for one of the tests.
We compute simple gaussian beams for :math:`\nu` and :math:`\nu_0 + \Delta \nu`, assuming a diffraction limited experiment.
This is thought to be used in the absence of data coming from the planets beams measurements.
"""

import numpy as np
from astropy import constants, units
import os
import tempfile
import yaml
from cobaya.install import install
from cobaya.model import get_model
from itertools import product

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

data_path = packages_path + "/data/MFLike/v0.8"

install({"likelihood": {"mflike.TTTEEE": None}}, path=packages_path)

cosmo_params = {
    "cosmomc_theta": 0.0104092,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

nuis_params = {
    "T_effd": 19.6,
    "beta_d": 1.5,
    "beta_s": -2.5,
    "alpha_s": 1,
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
    "cal_LAT_93": 1,
    "cal_LAT_145": 1,
    "cal_LAT_225": 1,
    "calG_all": 1,
    "alpha_LAT_93": 0,
    "alpha_LAT_145": 0,
    "alpha_LAT_225": 0,
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.20,
    "a_c": 4.90,
    "beta_c": 2.20,
    "a_s": 3.10,
    "T_d": 9.60,
    "a_gtt": 2.80,
    "xi": 0.10,
    "alpha_dT": -0.6,
    "alpha_p": 1,
    "alpha_tSZ": 0.,
    "calT_LAT_93": 1,
    "calT_LAT_145": 1,
    "calT_LAT_225": 1,
    "a_gte": 0.10,
    "a_pste": 0,
    "alpha_dE": -0.4,
    "a_gee": 0.10,
    "a_psee": 0,
    "alpha_dE": -0.4,
    "calE_LAT_93": 1,
    "calE_LAT_145": 1,
    "calE_LAT_225": 1,
}

info = {
    "likelihood": {
        "mflike.TTTEEE": {
            "input_file": "LAT_simu_sacc_00000.fits",
            "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
        },
    },
    "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
               "mflike.BandpowerForeground": None},
    "params": cosmo_params | nuis_params,
    "packages_path": packages_path,
}

model = get_model(info)
my_mflike = model.likelihood["mflike.TTTEEE"]

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

#first generating Gaussian beams for the frequencies in our arrays
beam_dict = {}
#generating the dictionary with (gaussian) beams for each nu+dnu
beam_dnu_dict = {}
dnu = np.arange(-20, 21, dtype = float)

for exp, spin in product(my_mflike.experiments, ["s0", "s2"]):
    nu0 = int(exp[4:])
    fwhm = compute_FWHM(nu0)
    key = f"{exp}_{spin}"
    nu = my_mflike.bands[key]['nu']
    
    beam_dict[key] = {"nu": nu, "beams": gauss_beams(fwhm, nu, nu0, 2, 10000, pol=spin=="s2")} 


    gbeambsh = gauss_beams(fwhm, nu0+dnu, nu0, 2, 10000, pol=spin=="s2")
    beam_dnu_dict[key] = {
            "beams": {f"{dn}": gbeambsh[idn] for idn, dn in enumerate(dnu)},
            "nu_0": nu0,
            "alpha": 2
            }

# saving the yaml file
with open(data_path + '/LAT_gauss_beams.yaml', 'w') as file:
    yaml.dump(beam_dict, file, default_flow_style=False)
    print("saving "+data_path + '/LAT_gauss_beams.yaml')


with open(data_path + '/LAT_beam_bandshift.yaml', 'w') as file:
    yaml.dump(beam_dnu_dict, file, default_flow_style=False)
    print("saving "+data_path + '/LAT_beam_bandshift.yaml')
