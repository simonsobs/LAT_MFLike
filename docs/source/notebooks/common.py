import numpy as np

cosmo_params = {
    "cosmomc_theta": 0.0104092,
    "As": 1e-10 * np.exp(3.044),
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

fg_params = {
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
    "a_psee": 0.003,
    "a_pste": 0.042,
    "xi": 0.10,
    "beta_s": -2.5,  # beta radio
    "alpha_s": 1,  # alpha radio
    "T_effd": 19.6,  # effective galactic dust temperature
    "beta_d": 1.5,  # beta galactic dust
    "alpha_dT": -0.6,  # galactic dust ell slope for T
    "alpha_dE": -0.4,  # galactic dust ell slope for E
    "alpha_tSZ": 0.0,  # tSZ ell slope
    "alpha_p": 1,  # CIB poisson ell slope
}

nuisance_params = {
    # only ideal values for now
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

mflike_input_file = dict(
    input_file="LAT_simu_sacc_00044.fits", cov_Bbl_file="data_sacc_w_covar_and_Bbl.fits"
)
mflike_config = {"mflike.TTTEEE": mflike_input_file}
