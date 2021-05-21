import numpy as np

class TheoryForge:

    def __init__(self,mflike):

        self.freqs = mflike.freqs
        self.foregrounds = mflike.foregrounds
        self.l_bpws = mflike.l_bpws
        self.requested_cls = mflike.requested_cls
        self.expected_params_fg = mflike.expected_params_fg
        self.expected_params_nuis = mflike.expected_params_nuis
        self.spec_meta  = mflike.spec_meta

        self._init_foreground_model()

        #Parameters for band integration
        self.bandint_nsteps = mflike.band_integration["nsteps"]
        self.bandint_width = mflike.band_integration["bandwidth"]


    def get_modified_theory(self,Dls,**params):
       
        fg_params = {k: params[k] for k in self.expected_params_fg}
        nuis_params = {k: params[k] for k in self.expected_params_nuis}


        #Bandpass construction for band integration
        if not hasattr(self.bandint_width, "__len__"):
            self.bandint_width = np.full_like(self.freqs,self.bandint_width,dtype=np.float)
        if np.any(np.array(self.bandint_width) > 0):
            assert self.bandint_nsteps > 1 , 'bandint_width and bandint_nsteps not coherent'
            assert np.all(np.array(self.bandint_width) > 0), 'one band has width = 0, set a positive width and run again'

            self.bandint_freqs = []
            for ifr,fr in enumerate(self.freqs):
                bandpar = 'bandint_shift_'+str(fr)
                bandlow = fr*(1-self.bandint_width[ifr]*.5)
                bandhigh = fr*(1+self.bandint_width[ifr]*.5)
                print(bandlow,bandhigh)
                nubtrue = np.linspace(bandlow,bandhigh,self.bandint_nsteps,dtype=float)
                nub = np.linspace(bandlow+nuis_params[bandpar],bandhigh+nuis_params[bandpar],self.bandint_nsteps,dtype=float)
                tranb = _cmb2bb(nub)
                tranb_norm = np.trapz(_cmb2bb(nubtrue),nubtrue)
                self.bandint_freqs.append([nub,tranb/tranb_norm])
        else:
            self.bandint_freqs = np.empty_like(self.freqs,dtype=float)
            for ifr,fr in enumerate(self.freqs):
                bandpar = 'bandint_shift_'+str(fr)
                self.bandint_freqs[ifr] = fr+nuis_params[bandpar]


        fg_dict = self._get_foreground_model(**fg_params)


        #Built theory 
        dls_dict = {}
        for m in self.spec_meta:
            p = m['pol']
            dls_dict[p,  m['nu1'], m['nu2']] = Dls[p] + fg_dict[p, 'all', m['nu1'], m['nu2']]

        return dls_dict



    def _init_foreground_model(self):

        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp
        
        #set pivot freq and multipole
        self.fg_nu_0 = self.foregrounds["normalisation"]["nu_0"]
        self.fg_ell_0 = self.foregrounds["normalisation"]["ell_0"]

        # We don't seem to be using this
        # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        self.cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
        self.cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

        components = self.foregrounds["components"]
        self.fg_component_list = {s: components[s] for s in self.requested_cls}



    def _get_foreground_model(self,**fg_params):

#        if not isinstance(self.freqs, np.ndarray):
#            frequencies = np.array(self.freqs)

        ell = self.l_bpws
        ell_0 = self.fg_ell_0
        nu_0 = self.fg_nu_0

        model = {}
        model["tt", "kSZ"] = fg_params["a_kSZ"] * self.ksz(
            {"nu": self.bandint_freqs},
            {"ell": ell, "ell_0": ell_0})
        model["tt", "cibp"] = fg_params["a_p"] * self.cibp(
            {"nu": self.bandint_freqs, "nu_0": nu_0,
            "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {"ell": ell, "ell_0": ell_0, "alpha": 2})
        model["tt", "radio"] = fg_params["a_s"] * self.radio(
            {"nu": self.bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2},
            {"ell": ell, "ell_0": ell_0, "alpha": 2})
        model["tt", "tSZ"] = fg_params["a_tSZ"] * self.tsz(
            {"nu": self.bandint_freqs, "nu_0": nu_0},
            {"ell": ell, "ell_0": ell_0})
        model["tt", "cibc"] = fg_params["a_c"] * self.cibc(
            {"nu": self.bandint_freqs, "nu_0": nu_0,
            "temp": fg_params["T_d"], "beta": fg_params["beta_c"]},
            {"ell": ell, "ell_0": ell_0, "alpha": 2 - fg_params["n_CIBC"]})

        fg_dict = {}
        for c1, f1 in enumerate(self.freqs):
            for c2, f2 in enumerate(self.freqs):
                for s in self.requested_cls:
                    fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
                        fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

        return fg_dict



def _cmb2bb(nu):
    # NB: numerical factors not included 
    from scipy import constants
    T_CMB = 2.72548
    x = nu*constants.h * 1e9 / constants.k / T_CMB
    return  np.exp(x)*(nu * x / np.expm1(x))**2

