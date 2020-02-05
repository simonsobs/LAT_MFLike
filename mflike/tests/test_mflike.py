import unittest
import os

modules_path = os.environ.get("COBAYA_MODULES") or "/tmp/modules"

cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544
}

nuisance_params= {
    "a_tSZ": 3.3044404448917724,
    "a_kSZ": 1.6646620740058649,
    "a_p": 6.912474322461401,
    "beta_p": 2.077474196171309,
    "a_c": 4.88617700670901,
    "beta_c": 2.2030316332596014,
    "n_CIBC": 1.20,
    "a_s": 3.099214100532393,
    "T_d": 9.60
}

chi2s = {
    "tt": {'pols':['TT'], 'chi2':490.4163, 'sym': True},
    "te": {'pols':['TE'], 'chi2':482.3090, 'sym': True},
    "ee": {'pols':['EE'], 'chi2':511.1752, 'sym': True},
    "tt-te-ee": {'pols':['TT','TE','EE'], 'chi2':1488.9766, 'sym': True}}

class MFLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install
        install({"likelihood": {"mflike.MFLike": None}}, path=modules_path)

    def test_mflike(self):
        import camb
        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v]
                   for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        for select, pars in chi2s.items():
            chi2 = pars['chi2']
            from mflike import MFLike
            my_mflike = MFLike({"path_install": modules_path,
                                "data_folder": "LAT_MFLike_data/like_products",
                                "input_file": "data_sacc_00000.fits",
                                "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                                "defaults":{"polarizations":pars['pols'],
                                            "scales":{"TT": [2, 6002],
                                                      "TE": [2, 6002],
                                                      "ET": [2, 6002],
                                                      "EE": [2, 6002]},
                                            "symmetrize": pars['sym']}})
            loglike = my_mflike.loglike(cl_dict, **nuisance_params)
            #self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)
            print(-2 * (loglike - my_mflike.logp_const))

    def test_cobaya(self):
        info = {"likelihood": {"mflike.MFLike": {"data_folder": "LAT_MFLike_data/like_products",
                                                 "input_file": "data_sacc_00000.fits",
                                                 "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                                                 "defaults":{"polarizations":['TT','TE','ET','EE'],
                                                             "scales":{"TT": [2, 6002],
                                                                       "TE": [2, 6002],
                                                                       "ET": [2, 6002],
                                                                       "EE": [2, 6002]},
                                                             "symmetrize": True}}},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": cosmo_params,
                "modules": modules_path}
        from cobaya.model import get_model
        model = get_model(info)
        my_mflike = model.likelihood["mflike.MFLike"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)
        #self.assertAlmostEqual(chi2[0], chi2s["tt-te-ee"]['chi2'], 2)
        print(chi2[0], chi2s["tt-te-ee"]['chi2'])
