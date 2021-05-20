import numpy as np

class TheoryForge:

    def __init__(self,mflike):

        self.freq = mflike.freq
        self.foregrounds = mflike.foregrounds
        self.l_bpws = mflike.l_bpws
        self.requested_cls = mflike.requested_cls
        self.expected_params_fg = mflike.expected_params_fg
        self.expected_params_nuis = mflike.expected_params_nuis
        self.spec_meta  = mflike.spec_meta


    def get_modified_theory(self,Dls,params):
       
        fg_params = {k: params[k] for k in self.expected_params_fg})            
        nuis_params = {k: params[k] for k in self.expected_params_nuis}) 

        fg_dict = self._get_foreground_model(fg_params)

        dls_dict = {}
        for m in self.spec_meta:
            p = m['pol']
            dls_dict[p,  m['nu1'], m['nu2']]) = Dls[p] + fg_model[p, 'all', m['nu1'], m['nu2']])

        return dls_dict

     def _get_foreground_model(self,fg_params):

         normalisation = self.foregrounds["normalisation"]
         nu_0 = normalisation["nu_0"]
         ell_0 = normalisation["ell_0"]

         from fgspectra import cross as fgc
         from fgspectra import frequency as fgf
         from fgspectra import power as fgp

         # We don't seem to be using this
         # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
         ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
         cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
         radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
         tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
         cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

         # Make sure to pass a numpy array to fgspectra
         if not isinstance(self.freqs, np.ndarray):
             frequencies = np.array(self.freqs)

         ell = self.l_bpws

         model = {}
         model["tt", "kSZ"] = fg_params["a_kSZ"] * ksz(
             {"nu": frequencies},
             {"ell": ell, "ell_0": ell_0})
         model["tt", "cibp"] = fg_params["a_p"] * cibp(
             {"nu": frequencies, "nu_0": nu_0,
             "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
             {"ell": ell, "ell_0": ell_0, "alpha": 2})
         model["tt", "radio"] = fg_params["a_s"] * radio(
             {"nu": frequencies, "nu_0": nu_0, "beta": -0.5 - 2},
             {"ell": ell, "ell_0": ell_0, "alpha": 2})
         model["tt", "tSZ"] = fg_params["a_tSZ"] * tsz(
             {"nu": frequencies, "nu_0": nu_0},
             {"ell": ell, "ell_0": ell_0})
         model["tt", "cibc"] = fg_params["a_c"] * cibc(
             {"nu": frequencies, "nu_0": nu_0,
             "temp": fg_params["T_d"], "beta": fg_params["beta_c"]},
             {"ell": ell, "ell_0": ell_0, "alpha": 2 - fg_params["n_CIBC"]})

         components = self.foregrounds["components"]
         component_list = {s: components[s] for s in self.requested_cls}
         fg_dict = {}
         for c1, f1 in enumerate(frequencies):
             for c2, f2 in enumerate(frequencies):
                 for s in self.requested_cls:
                     fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                     for comp in component_list[s]:
                         fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
                         fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

         return fg_dict
