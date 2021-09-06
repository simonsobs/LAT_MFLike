import os
import tempfile
import unittest
import sys
import numpy as np
import matplotlib.pyplot as plt

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(tempfile.gettempdir(), "plikmflike")

cosmo_params = {
	# Planck 2018 best fit parameters.
	# [arXiv:1807.06209]
	"cosmomc_theta" : 0.01040909,
	"As" : 2.10058e-9,
	"ns" : 0.96605,
	"ombh2" : 0.022383,
	"omch2" : 0.12011,
	"tau" : 0.0543
}

nuisance_params = {
	# TT parameters
	"A_cib_217" : 67.0,
	"cib_index" : -1.3,
	"xi_sz_cib" : 0.1,
	"A_sz" : 7.0,
	"ksz_norm" : 3.0,
	"gal545_A_100" : 7.0,
	"gal545_A_143" : 9.0,
	"gal545_A_143_217" : 21.0,
	"gal545_A_217" : 80.0,
	"ps_A_100_100" : 257.0,
	"ps_A_143_143" : 47.0,
	"ps_A_143_217" : 40.0,
	"ps_A_217_217" : 104.0,
	
	# TE parameters
	"galf_TE_index" : -2.4,
	"galf_TE_A_100" : 0.130,
	"galf_TE_A_100_143" : 0.130,
	"galf_TE_A_100_217" : 0.46,
	"galf_TE_A_143" : 0.207,
	"galf_TE_A_143_217" : 0.69,
	"galf_TE_A_217" : 1.938,
	
	# EE parameters
	"galf_EE_index" : -2.4,
	"galf_EE_A_100" : 0.055,
	"galf_EE_A_100_143" : 0.040,
	"galf_EE_A_100_217" : 0.094,
	"galf_EE_A_143" : 0.086,
	"galf_EE_A_143_217" : 0.21,
	"galf_EE_A_217" : 0.70,
	
	# calibration parameters
	"calib_100T" : 1.002,
	# T calibration at 143 GHz is fixed to 1.0
	"calib_217T" : 0.998,
	
	"calib_100P" : 1.021,
	"calib_143P" : 0.966,
	"calib_217P" : 1.040,
	"A_planck" : 1.000
}

# Calculated using the F90 plik code with the parameters above.
chi2_f90 = 2.0 * 1245.79402334843

class PlikMFLikeTest(unittest.TestCase):
	def setUp(self):
		from cobaya.install import install
		
		install({
			"likelihood" : {
				"plikmflike.PlikMFLike" : None
			}
		}, path = packages_path, skip_global = True)
	
	def test_plikmflike(self):
		import camb
		
		camb_params = cosmo_params.copy()
		camb_params.update({ "lmax" : 9000, "lens_potential_accuracy" : 1 })
		
		pars = camb.set_params(**camb_params)
		results = camb.get_results(pars)
		powers = results.get_cmb_power_spectra(pars, CMB_unit = "muK")
		cl_dict = { k : powers["total"][2:6500, v] for k, v in { "tt" : 0, "te" : 3, "ee" : 1 }.items() }
		cl_dict["ell"] = np.arange(2, 6500)
		
		from plikmflike import PlikMFLike
		
		plik = PlikMFLike({
			"packages_path" : packages_path,
			"data_folder" : "data/",
			"weightfile" : "bweight.dat",
			"minfile" : "blmin.dat",
			"maxfile" : "blmax.dat",
			"covfile" : "covmat.dat",
			"specfile" : "plikdata.dat",
			"leakfile" : "leakage.dat",
			"corrfile" : "ee_cnoise.dat",
			"subpixfile" : "subpix.dat",
			
			"lmin" : 30,
			"lmax_win" : 2508,
			"tt_lmax" : 3000
		})
		
		chi2 = -2.0 * plik.loglike(cl_dict, **nuisance_params)
		
		print("------------------------------------------")
		print("Calculated chi square value   : {:10.5f}".format(chi2))
		print("Chi square value from plik.f90: {:10.5f}".format(chi2_f90))
		print("------------------------------------------")
		

if __name__ == "__main__":
	unittest.main()
