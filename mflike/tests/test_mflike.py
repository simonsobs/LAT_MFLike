import os
import tempfile
import unittest
from cobaya.model import get_model
from cobaya.install import install

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

common_nuis_params = {
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
}

TT_nuis_params = {
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
}

TE_nuis_params = {
    "a_gte": 0.10,
    "a_pste": 0,
    "alpha_dE": -0.4,
}

EE_nuis_params = {
    "a_gee": 0.10,
    "a_psee": 0,
    "alpha_dE": -0.4,
    "calE_LAT_93": 1,
    "calE_LAT_145": 1,
    "calE_LAT_225": 1,
}

chi2s = {
    "tt": 920.2630056646808,
    "te-et": 1375.0253752060808,
    "ee": 850.9616194876838,
    "tt-ee": 1770.2453437088866,
    "tt-te-et": 2289.876148743457,
    "te-et-ee": 2221.3950392195356,
    "tt-te-et-ee": 3137.874002938912,
}
pre = "LAT_simu_sacc_"


class MFLikeTest(unittest.TestCase):
    def setUp(self):
        install({"likelihood": {"mflike.TTTEEE": None}}, path=packages_path)

    def test_mflike(self):
        from mflike import TTTEEE, BandpowerForeground
        import camb

        # using camb low accuracy parameters for the test
        camb_cosmo = cosmo_params | {"lmax": 9001, "lens_potential_accuracy": 1}
        pars = camb.set_params(**camb_cosmo)
        nuis_params = common_nuis_params | TT_nuis_params | TE_nuis_params | EE_nuis_params
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        for select, chi2 in chi2s.items():
            my_mflike = TTTEEE(
                {
                    "packages_path": packages_path,
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
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
            fg = BandpowerForeground(my_mflike.get_fg_requirements())
            fg_totals = fg.get_foreground_model_totals(**nuis_params)

            loglike = my_mflike.loglike(cl_dict, fg_totals, **nuis_params)
            self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)

    def test_cobaya_TT(self):
        nuis_params = common_nuis_params | TT_nuis_params
        nuis_params = {k: v for k, v in nuis_params.items() if "calE" not in k}
        info = {
            "likelihood": {
                "mflike.TT": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['tt']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TT"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["tt"], 2)

    def test_cobaya_TE(self):
        nuis_params = common_nuis_params | TE_nuis_params
        info = {
            "likelihood": {
                "mflike.TE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['te']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TE"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["te-et"], 2)

    def test_cobaya_EE(self):
        nuis_params = common_nuis_params | EE_nuis_params
        info = {
            "likelihood": {
                "mflike.EE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['ee']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
            "debug": True,
        }
        for _ in (False, True):
            model = get_model(info)
            my_mflike = model.likelihood["mflike.EE"]
            chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
            self.assertAlmostEqual(chi2, chi2s["ee"], 2)
            info["theory"].pop("mflike.BandpowerForeground", None)
            info["theory"]["mflike.EEForeground"] = None

    def test_cobaya_TT_EE(self):
        nuis_params = common_nuis_params | TT_nuis_params | EE_nuis_params
        info = {
            "likelihood": {
                "mflike.TTEE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['tt', 'ee']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TTEE"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["tt-ee"], 2)

    def test_cobaya_TT_TE(self):
        nuis_params = common_nuis_params | TT_nuis_params | TE_nuis_params
        info = {
            "likelihood": {
                "mflike.TTTE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['tt', 'te']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TTTE"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["tt-te-et"], 2)

    def test_cobaya_TE_EE(self):
        nuis_params = common_nuis_params | TE_nuis_params | EE_nuis_params
        info = {
            "likelihood": {
                "mflike.TEEE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground": {'requested_cls': ['te', 'ee']}},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TEEE"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["te-et-ee"], 2)

    def test_cobaya_TTTEEE(self):
        nuis_params = common_nuis_params | TT_nuis_params | TE_nuis_params | EE_nuis_params
        info = {
            "likelihood": {
                "mflike.TTTEEE": {
                    "input_file": pre + "00000.fits",
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
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        self.assertAlmostEqual(chi2, chi2s["tt-te-et-ee"], 2)

    def test_top_hat_bandpasses(self):
        nuis_params = common_nuis_params | TT_nuis_params | TE_nuis_params | EE_nuis_params
        # Let's vary values of bandint_shift parameters
        params = nuis_params | {
            k: {"prior": {"min": 0.9 * v, "max": 1.1 * v}}
            for k, v in nuis_params.items()
            if k.startswith("bandint_shift")
        }

        def _get_model(nsteps, bandwidth):
            info = {
                "likelihood": {
                    "mflike.TTTEEE": {
                        "input_file": pre + "00000.fits",
                        "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",

                    }
                },
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                           "mflike.BandpowerForeground": {"top_hat_band": {
                               "nsteps": nsteps,
                               "bandwidth": bandwidth,
                           }}},
                "params": {**cosmo_params, **params},
                "packages_path": packages_path,
            }

            _model = get_model(info)
            return _model, _model.likelihood["mflike.TTTEEE"].logp_const

        #  top hat band
        self.model1, logp_const = _get_model(nsteps=1, bandwidth=0)

        # 10 integrationn points and 10% of central frequency value
        self.model2, logp_const = _get_model(nsteps=10, bandwidth=0.1)

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
                chi2_mflike = -2 * (getattr(self, model).loglike(
                    new_params, return_derived=False) - logp_const)
                self.assertAlmostEqual(chi2_mflike, chi2[i], 1)

    def test_Gaussian_chromatic_beams(self):

        nuis_params = common_nuis_params | TT_nuis_params | TE_nuis_params | EE_nuis_params
        
        # generating the data products needed 
        test_path = os.path.dirname(__file__)
        import subprocess
        subprocess.run("python "+os.path.join(test_path, "../../scripts/generate_beams_w_bandpass_shifts.py"), shell=True, check=True)
        
        info = {
            "likelihood": {
                "mflike.TTTEEE": {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground":{
                           "beam_profile": {"beam_from_file": packages_path +
                                 "/data/MFLike/v0.8/LAT_gauss_beams.yaml"},
                }},
            "params": cosmo_params | nuis_params,
            "packages_path": packages_path,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood["mflike.TTTEEE"]
        chi2 = -2 * (model.loglike(nuis_params, return_derived=False) - my_mflike.logp_const)
        chi2s_beam = {"tt-te-et-ee": 4272.842504438564}
        self.assertAlmostEqual(chi2, chi2s_beam["tt-te-et-ee"], 2)


        model.close()
        
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

        info = {
                "likelihood": {
                    "mflike.TTTEEE": {
                        "input_file": pre + "00000.fits",
                        "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
                       },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}},
                       "mflike.BandpowerForeground":{
                          "beam_profile": {"beam_from_file": packages_path +
                                 "/data/MFLike/v0.8/LAT_gauss_beams.yaml",
                          "Bandpass_shifted_beams": packages_path + 
                                 "/data/MFLike/v0.8/LAT_beam_bandshift.yaml"},
                    },
                },
                "params": cosmo_params | params,
                "packages_path": packages_path,
            }

        model = get_model(info)
        logp_const = model.likelihood["mflike.TTTEEE"].logp_const

        chi2 = [4272.84250444, 10987.36734122, 127166.75296958]

        for i, bandshift in enumerate([0.0, 1.0, 5.0]):
            new_params = {
                    **params,
                    **{par: bandshift for par in params.keys() if par.startswith("bandint_shift")},
                }

            chi2_mflike = -2 * (model.loglike(new_params, return_derived=False) - logp_const)
            self.assertAlmostEqual(chi2_mflike, chi2[i], 2)
