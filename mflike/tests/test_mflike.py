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
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.08,
    "a_c": 4.90,
    "beta_c": 2.20,
    "n_CIBC": 1.20,
    "a_s": 3.10,
    "T_d": 9.60
}

chi2s = {
    "tt": 490.4163,
    "te": 482.3090,
    "ee": 511.1752,
    "tt-te-ee": 1488.9766}

class MFLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install
        install({"likelihood": {"mflike.MFLike": None}}, path=modules_path)

    def test_mflike(self):
        from mflike import MFLike
        my_mflike = MFLike({"path_install": modules_path, "sim_id": 0})
        import camb
        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v]
                   for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        for select, chi2 in chi2s.items():
            my_mflike = MFLike({"path_install": modules_path,
                                "sim_id": 0, "select": select})
            loglike = my_mflike.loglike(cl_dict, **nuisance_params)
            self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 3)

    def test_cobaya(self):
        info = {"likelihood": {"mflike.MFLike": {"sim_id": 0}},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": cosmo_params,
                "modules": modules_path}
        from cobaya.model import get_model
        model = get_model(info)
        my_mflike = model.likelihood["mflike.MFLike"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)
        self.assertAlmostEqual(chi2[0], chi2s["tt-te-ee"], 3)
