"""
.. module:: mflike

:Synopsis: Definition of simplistic likelihood for Simons Observatory
:Authors: Thibaut Louis, Xavier Garrido, Max Abitbol, Erminia Calabrese, Antony Lewis, David Alonso

"""
# Global
import os
import numpy as np

# Local
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.conventions import _path_install
from cobaya.likelihoods._base_classes import _InstallableLikelihood


class MFLike(_InstallableLikelihood):
    #install_options = {"github_repository": "simonsobs/LAT_MFLike_data",
    #                   "github_release": "v0.2"}
    install_options = {"download_url": "https://portal.nersc.gov/cfs/sobs/users/mflike_data_release/MFLike_data/data_sacc_v0.2.tar.gz"}

    def initialize(self):
        self.log.info("Initialising.")
        if not getattr(self, "path", None) and not getattr(self, "path_install", None):
            raise LoggedError(
                self.log, "No path given to MFLike data. Set the likelihood property "
                          "'path' or the common property '%s'.", _path_install)
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.path_install, "data"))

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log, "The 'data_folder' directory does not exist. "
                          "Check the given path [%s].", self.data_folder)

        self.prepare_data()

        # State requisites to the theory code
        self.requested_cls = ["tt", "te", "ee"]

        self.expected_params = ["a_tSZ", "a_kSZ", "a_p", "beta_p",
                                "a_c", "beta_c", "n_CIBC", "a_s", "T_d"]

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.", differences)

    def get_requirements(self):
        return dict(Cl={k: max(c, 9000) for k, c in self.lcuts.items()})

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=True)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        ps_vec = self._get_power_spectra(cl, **params_values)
        delta = self.data_vec - ps_vec
        logp = -0.5 * np.einsum('i,ij,j', delta, self.inv_cov, delta) + self.logp_const
        self.log.debug(
            "Log-likelihood value computed = {} (Χ² = {})".format(logp, -2 * logp))
        return logp

    def prepare_data(self, verbose=False):
        import sacc
        data = self.data
        input_fname = os.path.join(self.data_folder, self.input_file)
        s = sacc.Sacc.load_fits(input_fname)

        cbbl_extra = False
        s_b = s
        if self.cov_Bbl_file:
            if self.cov_Bbl_file != self.input_file:
                cov_Bbl_fname = os.path.join(self.data_folder, self.cov_Bbl_file)
                s_b = sacc.Sacc.load_fits(cov_Bbl_fname)
                cbbl_extra = True

        try:
            default_cuts = self.defaults
        except:
            raise KeyError('You must provide a list of default cuts')

        # Translation betwen TEB and sacc C_ell types
        pol_dict = {'T': '0',
                    'E': 'e',
                    'B': 'b'}
        ppol_dict = {'TT': 'tt',
                     'EE': 'ee',
                     'TE': 'te',
                     'ET': 'te',
                     'BB': 'bb',
                     'EB': 'eb',
                     'BE': 'eb',
                     'TB': 'tb',
                     'BT': 'tb',
                     'BB': 'bb'}

        def xp_nu(xp, nu):
            return xp + '_' + str(nu)

        def get_cl_meta(spec):
            exp_1, exp_2 = spec['experiments']
            freq_1, freq_2 = spec['frequencies']
            # Read off polarization channel combinations
            pols = spectrum.get('polarizations',
                                default_cuts['polarizations']).copy()
            # Read off scale cuts
            scls = spectrum.get('scales',
                                default_cuts['scales']).copy()

            # For the same two channels, do not include ET and TE, only TE
            if (exp_1 == exp_2) and (freq_1 == freq_2):
                if ('ET' in pols):
                    pols.remove('ET')
                    if ('TE' not in pols):
                        pols.append('TE')
                        scls['TE'] = scls['ET']
                symm = False
            else:
                # Symmetrization
                if ('TE' in pols) and ('ET' in pols):
                    symm = spectrum.get('symmetrize',
                                        default_cuts['symmetrize'])
                else:
                    symm = False

            return exp_1, exp_2, freq_1, freq_2, pols, scls, symm

        def get_sacc_names(pol, exp_1, exp_2, freq_1, freq_2):
            p1, p2 = pol
            tname_1 = xp_nu(exp_1, freq_1)
            tname_2 = xp_nu(exp_2, freq_2)
            if p1 in ['E', 'B']:
                tname_1 += '_s2'
            else:
                tname_1 += '_s0'
            if p2 in ['E', 'B']:
                tname_2 += '_s2'
            else:
                tname_2 += '_s0'
            dtype = 'cl_' + pol_dict[p1] + pol_dict[p2]
            return tname_1, tname_2, dtype

        # First trim the SACC file
        indices = []
        indices_b = []
        len_compressed = 0
        for spectrum in data['spectra']:
            exp_1, exp_2, freq_1, freq_2, pols, scls, symm = get_cl_meta(spectrum)
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2,
                                                         freq_1, freq_2)
                lmin, lmax = scls[pol]
                ind = s.indices(dtype,  # Select power spectrum type
                                (tname_1, tname_2),  # Select channel combinations
                                ell__gt=lmin, ell__lt=lmax)  # Scale cuts
                indices += list(ind)
                if cbbl_extra:
                    ind_b = s_b.indices(dtype,  # Select power spectrum type
                                        (tname_1, tname_2),  # Select channel combinations
                                        ell__gt=lmin, ell__lt=lmax)  # Scale cuts
                    indices_b += list(ind_b)
                if symm and pol == 'ET':
                    pass
                else:
                    len_compressed += ind.size

                if verbose:
                    print(tname_1, tname_2, dtype, ind.shape, lmin, lmax)
        # Get rid of all the unselected power spectra
        indices = np.array(indices)
        s.keep_indices(np.array(indices))
        if cbbl_extra:
            indices_b = np.array(indices_b)
            s_b.keep_indices(np.array(indices_b))

        # Now create metadata for each spectrum
        self.spec_meta = []
        len_full = s.mean.size
        mat_compress = np.zeros([len_compressed, len_full])
        mat_compress_b = np.zeros([len_compressed, len_full])
        bands = {}
        self.lcuts = {k: c[1] for k, c in default_cuts['scales'].items()}
        index_sofar = 0

        self.l_bpws = None
        for spectrum in data['spectra']:
            exp_1, exp_2, freq_1, freq_2, pols, scls, symm = get_cl_meta(spectrum)
            bands[xp_nu(exp_1, freq_1)] = freq_1
            bands[xp_nu(exp_2, freq_2)] = freq_2
            for k in scls.keys():
                self.lcuts[k] = max(self.lcuts[k], scls[k][1])
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2,
                                                         freq_1, freq_2)
                ind = s.indices(dtype,
                                (tname_1, tname_2))
                if cbbl_extra:
                    ind_b = s_b.indices(dtype,
                                        (tname_1, tname_2))

                if cbbl_extra:
                    ls, cls = s.get_ell_cl(dtype, tname_1, tname_2,
                                           return_windows=False)
                    _, _, ws = s_b.get_ell_cl(dtype, tname_1, tname_2,
                                              return_windows=True)
                else:
                    ls, cls, ws = s.get_ell_cl(dtype, tname_1, tname_2,
                                               return_windows=True)

                if self.l_bpws is None:
                    # The assumption here is that bandpower windows
                    # will all be sampled at the same ells.
                    self.l_bpws = ws[0]

                if (pol in ['TE', 'ET']) and symm:
                    pol2 = pol[::-1]
                    pols.remove(pol2)
                    tname_1, tname_2, dtype = get_sacc_names(pol2, exp_1, exp_2,
                                                             freq_1, freq_2)
                    ind2 = s.indices(dtype,
                                     (tname_1, tname_2))
                    cls2 = s.get_ell_cl(dtype, tname_1, tname_2)[1]
                    cls = 0.5 * (cls + cls2)

                    for i, (j1, j2) in enumerate(zip(ind, ind2)):
                        mat_compress[index_sofar + i, j1] = 0.5
                        mat_compress[index_sofar + i, j2] = 0.5
                    if cbbl_extra:
                        ind2_b = s_b.indices(dtype,
                                             (tname_1, tname_2))
                        for i, (j1, j2) in enumerate(zip(ind_b, ind2_b)):
                            mat_compress_b[index_sofar + i, j1] = 0.5
                            mat_compress_b[index_sofar + i, j2] = 0.5
                else:
                    for i, j1 in enumerate(ind):
                        mat_compress[index_sofar + i, j1] = 1
                    if cbbl_extra:
                        for i, j1 in enumerate(ind_b):
                            mat_compress_b[index_sofar + i, j1] = 1
                self.spec_meta.append({'ids': index_sofar + np.arange(cls.size, dtype=int),
                                       'pol': ppol_dict[pol],
                                       't1': xp_nu(exp_1, freq_1),
                                       't2': xp_nu(exp_2, freq_2),
                                       'nu1': freq_1,
                                       'nu2': freq_2,
                                       'leff': ls,
                                       'cl_data': cls,
                                       'bpw': ws})
                index_sofar += cls.size
        if not cbbl_extra:
            mat_compress_b = mat_compress
        self.data_vec = np.dot(mat_compress,s.mean)
        self.cov = np.dot(mat_compress_b,
                          s_b.covariance.covmat.dot(mat_compress_b.T))
        self.inv_cov = np.linalg.inv(self.cov)
        self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2) - \
                          0.5 * np.linalg.slogdet(self.cov)[1]


        # TODO: we should actually be using bandpass integration
        self.bands = sorted(bands)
        self.freqs = np.array([bands[b] for b in self.bands])

        self.lcuts = {k.lower(): c for k, c in self.lcuts.items()}
        if 'et' in self.lcuts:
            del self.lcuts['et']

    def _get_power_spectra(self, cl, **params_values):
        # Get Cl's from the theory code
        Dls = {s: cl[s][self.l_bpws] for s, _ in self.lcuts.items()} 

        # Get new foreground model given its nuisance parameters
        fg_model = self._get_foreground_model(
            {k: params_values[k] for k in self.expected_params})

        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m['pol']
            i = m['ids']
            w = m['bpw'][1]
            clt = np.dot(w, Dls[p] + fg_model[p, 'all', m['nu1'], m['nu2']])
            ps_vec[i] = clt

        return ps_vec

    def _get_foreground_model(self, fg_params):
        # Might change given different lmax
        l = self.l_bpws

        foregrounds = self.foregrounds
        normalisation = foregrounds["normalisation"]
        nu_0 = normalisation["nu_0"]
        ell_0 = normalisation["ell_0"]
        T_CMB = normalisation["T_CMB"]

        from fgspectra import cross as fgc
        from fgspectra import power as fgp
        from fgspectra import frequency as fgf
        cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
        cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

        model = {}
        model["tt", "kSZ"] = fg_params["a_kSZ"] * ksz(
            {"nu": self.freqs},
            {"ell": l, "ell_0": ell_0})
        model["tt", "cibp"] = fg_params["a_p"] * cibp(
            {"nu": self.freqs, "nu_0": nu_0, "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {"ell": l, "ell_0": ell_0, "alpha": 2})
        model["tt", "radio"] = fg_params["a_s"] * radio(
            {"nu": self.freqs, "nu_0": nu_0, "beta": -0.5 - 2},
            {"ell": l, "ell_0": ell_0, "alpha": 2})
        model["tt", "tSZ"] = fg_params["a_tSZ"] * tsz(
            {"nu": self.freqs, "nu_0": nu_0},
            {"ell": l, "ell_0": ell_0})
        model["tt", "cibc"] = fg_params["a_c"] * cibc(
            {"nu": self.freqs, "nu_0": nu_0, "temp": fg_params["T_d"], "beta": fg_params["beta_c"]},
            {"ell": l, "ell_0": ell_0, "alpha": 2 - fg_params["n_CIBC"]})

        components = foregrounds["components"]
        component_list = {s: components[s] for s in self.requested_cls}
        fg_model = {}
        for c1, f1 in enumerate(self.freqs):
            for c2, f2 in enumerate(self.freqs):
                for s in self.requested_cls:
                    fg_model[s, "all", f1, f2] = np.zeros(len(l))
                    for comp in component_list[s]:
                        fg_model[s, comp, f1, f2] = model[s, comp][c1, c2]
                        fg_model[s, "all", f1, f2] += fg_model[s, comp, f1, f2]

        return fg_model
