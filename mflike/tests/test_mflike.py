import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

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
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.20,
    "a_c": 4.90,
    "beta_c": 2.20,
    "a_s": 3.10,
    "T_d": 9.60,
    "a_gtt": 2.80,
    "a_gte": 0.10,
    "a_gee": 0.10,
    "a_psee": 0,
    "a_pste": 0,
    "xi": 0.10,
    "beta_s": -2.5,    
    "alpha_s": 1,      
    "T_effd": 19.6,    
    "beta_d": 1.5,     
    "alpha_dT": -0.6,  
    "alpha_dE": -0.4,  
    "alpha_p": 1, 
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
   # "calT_LAT_93": 1,
    "calE_LAT_93": 1,
   # "calT_LAT_145": 1,
    "calE_LAT_145": 1,
   # "calT_LAT_225": 1,
    "calE_LAT_225": 1,
    "cal_LAT_93": 1,
    "cal_LAT_145": 1,
    "cal_LAT_225": 1,
   # "calG_all": 1,
   # "alpha_LAT_93": 0,
   # "alpha_LAT_145": 0,
   # "alpha_LAT_225": 0,
}

chi2s = {
    "tt": 920.2630056646808,
    "te-et": 1375.0253752060808,
    "ee": 850.9616194876838,
    "tt-te-et-ee": 3137.874002938912,
}
pre = "LAT_simu_sacc_"


class MFLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"mflike.MFLike": None}}, path=packages_path)

    def test_mflike(self):
        import camb

        camb_cosmo = cosmo_params.copy()
        #using camb low accuracy parameters for the test
        camb_cosmo.update({"lmax": 9001, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        for select, chi2 in chi2s.items():
            from mflike import MFLike

            my_mflike = MFLike(
                {
                    "packages_path": packages_path,
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file":  "data_sacc_w_covar_and_Bbl.fits",
                    "defaults": {
                        "polarizations": select.upper().split("-"),
                        "scales": {
                            "TT": [30, 9000],
                            "TE": [30, 9000],
                            "ET": [30, 9000],
                            "EE": [30, 9000],
                        },
                        "symmetrize": False,
                    },
                }
            )
            loglike = my_mflike.loglike(cl_dict,  **nuis_params)
            self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)

    def test_cobaya(self):
        info = {
            "likelihood": {
                "mflike.MFLike": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                    },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": cosmo_params,
            "packages_path": packages_path,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood["mflike.MFLike"]
        chi2 = -2 * (model.loglikes(nuis_params)[0] - my_mflike.logp_const)
        self.assertAlmostEqual(chi2[0], chi2s["tt-te-et-ee"], 2)

    def test_top_hat_bandpasses(self):
        from copy import deepcopy

        # Let's vary values of bandint_shift parameters
        params = deepcopy(nuis_params)
        params.update(
            {
                k: {"prior": dict(min=0.9 * v, max=1.1 * v)}
                for k, v in params.items()
                if k.startswith("bandint_shift")
            }
        )

        def get_model(nsteps, bandwidth):
            info = {
                "likelihood": {
                    "mflike.MFLike": {
                        "input_file": pre + "00000.fits",
                        "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                        "top_hat_band": {
                            "nsteps": nsteps,
                            "bandwidth": bandwidth,
                        },
                    }
                },
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params,  **params},
                "packages_path": packages_path,
            }

            from cobaya.model import get_model

            model = get_model(info)
            return model, model.likelihood["mflike.MFLike"].logp_const

        #  top hat band
        self.model1, logp_const = get_model(nsteps=1, bandwidth=0)

        # 10 integrationn points and 10% of central frequency value
        self.model2, logp_const = get_model(nsteps=10, bandwidth=0.1)

        # chi2 reference results for the different models and different bandshifts
        chi2s = {
            "model1": [4614.10014353, 4929.98648522, 92894.01666485],
            "model2": [3734.34320513, 6034.08286905, 103597.03300165],
        }

        for model, chi2 in chi2s.items():
            for i, bandshift in enumerate([0.0, 1.0, 5.0]):
                new_params = {
                    **params,
                    **{par: bandshift for par in params.keys() if par.startswith("bandint_shift")},
                }
                chi2_mflike = -2 * (getattr(self, model).loglikes(new_params)[0] - logp_const)
                self.assertAlmostEqual(chi2_mflike[0], chi2[i], 2)
