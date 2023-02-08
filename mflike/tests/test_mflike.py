import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

nuisance_params = {
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.08,
    "a_c": 4.90,
    "beta_c": 2.20,
    "a_s": 3.10,
    "T_d": 9.60,
    "a_gtt": 2.81,
    "a_gte": 0.10,
    "a_gee": 0.10,
    "a_psee": 0,
    "a_pste": 0,
    "xi": 0.20,
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
    "calT_LAT_93": 1,
    "calE_LAT_93": 1,
    "calT_LAT_145": 1,
    "calE_LAT_145": 1,
    "calT_LAT_225": 1,
    "calE_LAT_225": 1,
    "cal_LAT_93": 1,
    "cal_LAT_145": 1,
    "cal_LAT_225": 1,
    "calG_all": 1,
    "alpha_LAT_93": 0,
    "alpha_LAT_145": 0,
    "alpha_LAT_225": 0,
}

chi2s = {
    "tt":  630.67697,
    "te-et": 935.7031,
    "ee": 729.74398,
    "tt-te-et-ee": 2287.6825,
}
pre = "data_sacc_"


class MFLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"mflike.MFLike": None}}, path=packages_path)

    def test_mflike(self):
        import camb

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
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
                    "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
                    "defaults": {
                        "polarizations": select.upper().split("-"),
                        "scales": {
                            "TT": [2, 5000],
                            "TE": [2, 5000],
                            "ET": [2, 5000],
                            "EE": [2, 5000],
                        },
                        "symmetrize": False,
                    },
                }
            )
            loglike = my_mflike.loglike(cl_dict, **nuisance_params)
            self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)

    def test_cobaya(self):
        info = {
            "likelihood": {
                "mflike.MFLike": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
                }
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": cosmo_params,
            "packages_path": packages_path,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood["mflike.MFLike"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)
        self.assertAlmostEqual(chi2[0], chi2s["tt-te-et-ee"], 2)

    def test_top_hat_bandpasses(self):
        from copy import deepcopy

        # Let's vary values of bandint_shift parameters
        params = deepcopy(nuisance_params)
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
                        "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
                        "top_hat_band": {
                            "nsteps": nsteps,
                            "bandwidth": bandwidth,
                        },
                    }
                },
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **params},
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
        chi2s = {"model1": [2287.684,3766.519,43787.995], "model2": [2368.41,4248.328,41804.223]}

        for model, chi2 in chi2s.items():
            for i, bandshift in enumerate([0.0, 1.0, 5.0]):
                new_params = {
                    **params,
                    **{par: bandshift for par in params.keys() if par.startswith("bandint_shift")},
                }
                chi2_mflike = -2 * (getattr(self, model).loglikes(new_params)[0] - logp_const)
                self.assertAlmostEqual(chi2_mflike[0], chi2[i], 2)
