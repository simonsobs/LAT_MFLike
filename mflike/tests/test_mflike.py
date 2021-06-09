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
    "a_tSZ": 3.3044404448917724,
    "a_kSZ": 1.6646620740058649,
    "a_p": 6.912474322461401,
    "beta_p": 2.077474196171309,
    "a_c": 4.88617700670901,
    "beta_c": 2.2030316332596014,
    "n_CIBC": 1.20,
    "a_s": 3.099214100532393,
    "T_d": 9.60,
    "a_gtt": 2.819845541,
    "a_gte": 0.081234536,
    "a_gee": 0.063235678,
    "a_psee": 0.003123453,
    "a_pste": 0.042312344,
    "xi": 0.03,
}

chi2s = {"tt": 1384.8184, "te": 1400.4507, "ee": 1428.8454, "tt-te-et-ee": 2413.1038}
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
            "modules": packages_path,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood["mflike.MFLike"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)
        self.assertAlmostEqual(chi2[0], chi2s["tt-te-et-ee"], 2)
