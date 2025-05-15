r"""
.. module:: mflike

:Synopsis: Definition of likelihood for Simons Observatory
:Authors: Simons Observatory Collaboration PS Group

MFLike is a multi frequency likelihood code that can be interfaced with the Cobaya
sampler and a theory Boltzmann code such as CAMB, CLASS or Cosmopower.

The ``MFLike`` likelihood class reads the data file (in ``sacc`` format)
and all the settings
for the MCMC run (such as file paths, :math:`\ell` ranges, experiments
and frequencies to be used, parameters priors...) from the ``MFLike.yaml`` file.

The theory :math:`C_{\ell}` are then summed with the (possibly frequency
integrated) foreground power spectra from the ``BandpowerForeground`` class,
and modified by systematic effects and calibrations.
The underlying foreground spectra are computed through ``fgspectra``.


This class applies four kinds of systematic effects to the CMB + foreground power spectrum:
    * calibrations (global ``calG_all``, per channel ``cal_exp``, per field
      ``calT_exp``, ``calE_exp``)
    * polarization angles effect (``alpha_exp``)
    * beam chromaticity (i.e. integrating the foreground SEDs with frequency dependent
      beams)
    * systematic templates (e.g. T --> P leakage). In this case the dictionary
      ``systematics_template`` has to be filled with the correct path
      ``rootname``:

      .. code-block:: yaml

        systematics_template:
          rootname: "test_template"

If left ``null``, no systematic template is applied.

The values of the systematic parameters are set in the ``TTTEEE/TTTE/TT/EE/TE/etc.yaml`` files corresponding to the classes that inherit the ``_MFLike`` one.  They have to be named as
``cal/calT/calE/alpha`` + ``_`` + experiment_channel string (e.g. ``LAT_93/dr6_pa4_f150``).
"""

import os
from typing import Optional
import numpy as np
from numbers import Real
import sacc
from cobaya.conventions import data_path, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

class _MFLike(InstallableLikelihood):
    _url = "https://portal.nersc.gov/cfs/sobs/users/MFLike_data"
    _release = "v0.8"
    install_options = {
        "download_url": f"{_url}/{_release}.tar.gz",
        "data_path": "MFLike",
    }

    # attributes set from .yaml
    input_file: Optional[str]
    cov_Bbl_file: Optional[str]
    data_folder: str
    data: dict
    defaults: dict
    systematics_template: dict
    supported_params: dict
    lmax_theory: Optional[int]
    requested_cls: list[str]

    def initialize(self):
        # Set default values to data member not initialized via yaml file
        self.l_bpws = None
        self.spec_meta = []

        # Set path to data
        if not getattr(self, "path", None) and not getattr(self, packages_path_input, None):
            raise LoggedError(
                self.log,
                "No path given to MFLike data. Set the likelihood property "
                f"'path' or the common property '{packages_path_input}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, data_path)
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. "
                f"Check the given path [{self.data_folder}].",
            )

        # Read data
        self._prepare_data()

        self.lmax_theory = self.lmax_theory or 9000
        self.log.debug(f"Maximum multipole value: {self.lmax_theory}")

        if self.systematics_template:
            # Initialize template for marginalization, if needed
            self._init_template_from_file()

        self._constant_nuisance: Optional[dict] = None
        self.log.info("Initialized!")

    def get_fg_requirements(self):
        return {"ells": self.l_bpws,
                "requested_cls": self.requested_cls,
                "experiments": self.experiments,
                "bands": self.bands,
                "beams": self.beams}

    def get_requirements(self):
        r"""
        Gets the foreground dictionary and theory :math:`D_{\ell}` from the Boltzmann solver code used,
        for the :math:`\ell` range up to the :math:`\ell_{max}` specified in the yaml

        :return: the dictionary of theory :math:`D_{\ell}` and foregrounds
        """

        return {"fg_totals": self.get_fg_requirements(),
                "Cl": {k: max(c, self.lmax_theory + 1) for k, c in self.lcuts.items()}}

    def logp(self, **params_values):
        cl = self.provider.get_Cl(ell_factor=True)
        fg_totals = self.provider.get_fg_totals()
        return self._loglike(cl, fg_totals, params_values)

    def _loglike(self, cl, fg_totals, params_values):
        r"""
        Computes the gaussian log-likelihood

        :param cl: the dictionary of theory + foregrounds :math:`D_{\ell}`
        :param fg_totals: the dictionary of foreground arrays
        :param params_values: the dictionary of all foreground + systematic parameters

        :return: the exact loglikelihood :math:`\ln \mathcal{L}`
        """
        ps_vec = self._get_power_spectra(cl, fg_totals, **params_values)
        delta = self.data_vec - ps_vec
        # logp = -0.5 * (delta @ self.inv_cov @ delta)
        chi2 = self._fast_chi_squared(self.inv_cov, delta)
        logp = -0.5 * chi2 + self.logp_const
        self.log.debug(f"Log-likelihood value computed = {logp} (Χ² = {chi2})")
        return logp

    def loglike(self, cl, fg_totals, **params_values):
        r"""
        Computes the gaussian log-likelihood, callable independent of Cobaya.

        :param cl: the dictionary of theory + foregrounds :math:`D_{\ell}`
        :param fg_totals: the dictionary of foreground arrays, can be obtained from ``BandpowerForeground``
        :param params_values: the dictionary of required foreground + systematic parameters

        :return: the exact loglikelihood :math:`\ln \mathcal{L}`
        """
        # This is needed if someone calls the function without initializing the likelihood
        # (typically a call with a precomputed Cl and no cobaya initialization steps e.g.
        # test_mflike)
        if self._constant_nuisance is None:
            from cobaya.parameterization import expand_info_param
            # pre-set default nuisance parameters
            self._constant_nuisance = {p: float(v) for p, info in self.params.items()
                                       if isinstance(v := expand_info_param(info).get("value"), Real)}

        params_values = self._constant_nuisance | params_values

        return self._loglike(cl, fg_totals, params_values)

    def _prepare_data(self):
        r"""
        Reads the sacc data, extracts the data tracers,
        trims the spectra and covariance according to the :math:`\ell` scales
        set in the input file, inverts the covariance, extracts bandpass info from
        the sacc file.
        It fills the list ``self.spec_meta`` (used throughout the code) of
        dictionaries with info about polarization, arrays combination, :math:`\ell`
        range, bandpowers and :math:`D_{\ell}` for each power spectrum required
        in the yaml.
        """
        data = self.data
        # Read data
        input_fname = os.path.normpath(os.path.join(self.data_folder, self.input_file))
        s = sacc.Sacc.load_fits(input_fname)

        # Read extra file containing covariance and windows if needed.
        cbbl_extra = False
        s_b = s
        if self.cov_Bbl_file and self.cov_Bbl_file != self.input_file:
            cov_Bbl_fname = os.path.join(self.data_folder, self.cov_Bbl_file)
            s_b = sacc.Sacc.load_fits(cov_Bbl_fname)
            cbbl_extra = True

        try:
            default_cuts = self.defaults
        except AttributeError:
            raise KeyError("You must provide a list of default cuts")

        # Translation between TEB and sacc C_ell types
        pol_dict = {"T": "0", "E": "e", "B": "b"}
        ppol_dict = {
            "TT": "tt",
            "EE": "ee",
            "TE": "te",
            "ET": "te",
            "BB": "bb",
            "EB": "eb",
            "BE": "eb",
            "TB": "tb",
            "BT": "tb",
        }

        def get_cl_meta(spec):
            """
            Lower-level function of `prepare_data`.
            For each of the entries of the `spectra` section of the
            yaml file, extracts the relevant information: channel,
            polarization combinations, scale cuts and
            whether TE should be symmetrized.

            :param spec: the dictionary ``data["spectra"]``
            """
            exp_1, exp_2 = spec["experiments"]
            # Read off polarization channel combinations
            pols = spec.get("polarizations", default_cuts["polarizations"]).copy()
            # Read off scale cuts
            scls = spec.get("scales", default_cuts["scales"]).copy()

            # For the same two channels, do not include ET and TE, only TE
            if exp_1 == exp_2:
                if "ET" in pols:
                    pols.remove("ET")
                    if "TE" not in pols:
                        pols.append("TE")
                        scls["TE"] = scls["ET"]
                symm = False
            else:
                # Symmetrization
                if ("TE" in pols) and ("ET" in pols):
                    symm = spec.get("symmetrize", default_cuts["symmetrize"])
                else:
                    symm = False

            return exp_1, exp_2, pols, scls, symm

        def get_sacc_names(pol, exp_1, exp_2):
            r"""
            Lower-level function of `prepare_data`.
            Translates the polarization combination and channel
            name of a given entry in the `spectra`
            part of the input yaml file into the names expected
            in the SACC files.

            :param pol: temperature or polarization fields, i.e. 'TT', 'TE'
            :param exp_1: frequency array of map 1
            :param exp_2: frequency array of map 2

            :return: tracer name 1, tracer name 2, string for :math:`C_{\ell}`
                     type (whether temperature or polarization)
            """
            tname_1 = exp_1
            tname_2 = exp_2
            p1, p2 = pol
            if p1 in ["E", "B"]:
                tname_1 += "_s2"
            else:
                tname_1 += "_s0"
            if p2 in ["E", "B"]:
                tname_2 += "_s2"
            else:
                tname_2 += "_s0"

            if p2 == "T":
                dtype = "cl_" + pol_dict[p2] + pol_dict[p1]
            else:
                dtype = "cl_" + pol_dict[p1] + pol_dict[p2]
            return tname_1, tname_2, dtype

        # First we trim the SACC file so it only contains
        # the parts of the data we care about.
        # Indices to be kept
        indices = []
        indices_b = []
        # Length of the final data vector
        len_compressed = 0
        for spectrum in data["spectra"]:
            exp_1, exp_2, pols, scls, symm = get_cl_meta(spectrum)
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2)
                lmin, lmax = scls[pol]
                ind = s.indices(
                    dtype,  # Power spectrum type
                    (tname_1, tname_2),  # Channel combinations
                    ell__gt=lmin,
                    ell__lt=lmax,
                )  # Scale cuts
                indices += list(ind)

                # Note that data in the cov_Bbl file may be in different order.
                if cbbl_extra:
                    ind_b = s_b.indices(dtype, (tname_1, tname_2), ell__gt=lmin, ell__lt=lmax)
                    indices_b += list(ind_b)

                if symm and pol == "ET":
                    pass
                else:
                    len_compressed += ind.size

                self.log.debug(f"{tname_1} {tname_2} {dtype} {ind.shape} {lmin} {lmax}")

        #The following is needed for soliket to trim cross-covariance
        if cbbl_extra:
            self.indices_soliket = np.zeros(s_b.mean.size, dtype=bool)
            self.indices_soliket[indices_b] = True
        else:
            self.indices_soliket = np.zeros(s.mean.size, dtype=bool)
            self.indices_soliket[indices] = True

        # Get rid of all the unselected power spectra.
        # Sacc takes care of performing the same cuts in the
        # covariance matrix, window functions etc.
        s.keep_indices(np.array(indices))
        if cbbl_extra:
            s_b.keep_indices(np.array(indices_b))

        # Now create metadata for each spectrum
        len_full = s.mean.size
        # These are the matrices we'll use to compress the data if
        # `symmetrize` is true.
        # Note that a lot of the complication in this function is caused by the
        # symmetrization option, for which SACC doesn't have native support.
        mat_compress = np.zeros([len_compressed, len_full])
        mat_compress_b = np.zeros([len_compressed, len_full])

        self.lcuts = {k: c[1] for k, c in default_cuts["scales"].items()}
        index_sofar = 0

        for spectrum in data["spectra"]:
            exp_1, exp_2, pols, scls, symm = get_cl_meta(spectrum)
            for k in scls.keys():
                self.lcuts[k] = max(self.lcuts[k], scls[k][1])
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2)
                # The only reason why we need indices is the symmetrization.
                # Otherwise all of this could have been done in the previous
                # loop over data["spectra"].
                ls, cls, ind = s.get_ell_cl(dtype, tname_1, tname_2, return_ind=True)
                if cbbl_extra:
                    ind_b = s_b.indices(dtype, (tname_1, tname_2))
                    ws = s_b.get_bandpower_windows(ind_b)
                else:
                    ws = s.get_bandpower_windows(ind)
                # pre-compute the actual slices of the weights that are needed
                nonzeros = np.array([np.nonzero(ws.weight[:, i])[0][[0, -1]] for i in range(ws.weight.shape[1])])
                ws.nonzeros = [slice(i[0], i[1] + 1) for i in nonzeros]
                ws.sliced_weights = [np.ascontiguousarray(ws.weight[ws.nonzeros[i], i])
                                     for i in range(len(nonzeros))]

                if self.l_bpws is None:
                    # The assumption here is that bandpower windows
                    # will all be sampled at the same ells.
                    self.l_bpws = ws.values

                # Symmetrize if needed. If symmetrize = True, the "ET" polarization
                # is eliminated by the polarization list and the TE spectrum becomes
                # (TE + ET)/2. The associated spec_meta dict will have "hasYX_xsp": False
                if (pol in ["TE", "ET"]) and symm:
                    pol2 = pol[::-1]
                    pols.remove(pol2)
                    tname_1, tname_2, dtype = get_sacc_names(pol2, exp_1, exp_2)
                    ind2 = s.indices(dtype, (tname_1, tname_2))
                    cls2 = s.get_ell_cl(dtype, tname_1, tname_2)[1]
                    cls = 0.5 * (cls + cls2)

                    for i, (j1, j2) in enumerate(zip(ind, ind2)):
                        mat_compress[index_sofar + i, j1] = 0.5
                        mat_compress[index_sofar + i, j2] = 0.5
                    if cbbl_extra:
                        ind2_b = s_b.indices(dtype, (tname_1, tname_2))
                        for i, (j1, j2) in enumerate(zip(ind_b, ind2_b)):
                            mat_compress_b[index_sofar + i, j1] = 0.5
                            mat_compress_b[index_sofar + i, j2] = 0.5
                else:
                    for i, j1 in enumerate(ind):
                        mat_compress[index_sofar + i, j1] = 1
                    if cbbl_extra:
                        for i, j1 in enumerate(ind_b):
                            mat_compress_b[index_sofar + i, j1] = 1
                # The fields marked with # below aren't really used, but
                # we store them just in case.
                self.spec_meta.append(
                    {
                        "ids": (index_sofar + np.arange(cls.size, dtype=int)),
                        "pol": ppol_dict[pol],
                        # this flag is true for pol = ET, BE, BT
                        "hasYX_xsp": pol in ["ET", "BE", "BT"],
                        "t1": exp_1,
                        "t2": exp_2,
                        "leff": ls,  #
                        "cl_data": cls,  #
                        "bpw": ws,
                    }
                )
                index_sofar += cls.size
        if not cbbl_extra:
            mat_compress_b = mat_compress
        # Put data and covariance in the right order.
        self.data_vec = np.dot(mat_compress, s.mean)
        self.cov = np.dot(mat_compress_b, s_b.covariance.covmat.dot(mat_compress_b.T))
        self.inv_cov = np.linalg.inv(self.cov)
        self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.cov)[1]

        self.experiments = data["experiments"]
        self.bands = {}
        self.beams = {}
        for name, tracer in s.tracers.items():
            self.bands[name] = {"nu": tracer.nu, "bandpass": tracer.bandpass}
            # trying to read beams, if present, and check if it is empty
            if hasattr(tracer, "beam") and np.size(tracer.beam) != 0:
                # transposing the beam since it is (nells, nfreqs) in sacc
                self.beams[name] = {"nu": tracer.nu, "beams": tracer.beam.T }
        
        # Put lcuts in a format that is recognisable by CAMB.
        self.lcuts = {k.lower(): c for k, c in self.lcuts.items()}
        if "et" in self.lcuts:
            del self.lcuts["et"]

        self.log.info(f"Number of bins used: {self.data_vec.size}")

    def _get_power_spectra(self, cl, fg_totals, **params_values):
        r"""
        Gets the theory :math:`D_{\ell}`, adds foregrounds :math:`D_{\ell}`
        and applies possible systematic effects through the ``get_modified_theory``
        function from the ``BandpowerForeground`` class. The spectra get then binned
        like the data.

        :param cl: the dictionary of theory :math:`D_{\ell}`
        :param fg_totals: the dictionary of foreground arrays
        :param params_values_nocosmo: the dictionary of required foregrounds
                                      and systematics parameters

        :return: the binned data vector
        """
        dls = {s: cl[s][self.l_bpws] for s, _ in self.lcuts.items()}
        dls_obs = self.get_modified_theory(dls, fg_totals, **params_values)

        return self._get_ps_vec(dls_obs)

    def _get_ps_vec(self, DlsObs):
        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m["pol"]
            w = m["bpw"]
            # If symmetrize = False, the (ET, exp1, exp2) spectrum
            # will have the flag m["hasYX_xsp"] = True.
            # In this case, the power spectrum
            # is computed as DlsObs["te", m["t2"], m["t1"]], to associate
            # T --> exp2, E --> exp1
            dls_obs = DlsObs[p, m["t2"], m["t1"]] if m["hasYX_xsp"] else DlsObs[p, m["t1"], m["t2"]]

            for i, nonzero, weights in zip(m["ids"], w.nonzeros, w.sliced_weights):
                ps_vec[i] = weights @ dls_obs[nonzero]
            # can check against unoptimized version
            # assert np.allclose(ps_vec[m["ids"]], np.dot(w.weight.T, dls_obs))
        return ps_vec

    def get_modified_theory(self, Dls, fg_totals, **nuis_params):
        r"""
        Takes the theory :math:`D_{\ell}`, sums it to the total
        foreground power spectrum (possibly computed with bandpass
        shift and bandpass integration) computed by ``_get_foreground_model``
        and applies calibration,
        polarization angles rotation and systematic templates.

        :param Dls: CMB theory spectra
        :param fg_totals: dictionary of foreground spectra
        :param nuis_params: dictionary of nuisance and foregrounds parameters

        :return: the CMB+foregrounds :math:`D_{\ell}` dictionary,
                 modulated by systematics
        """

        cmbfg_dict = {(s, exp1, exp2): Dls[s] + total_fg[i, j]  # Sum CMB and FGs
                      for i, exp1 in enumerate(self.experiments)
                      for j, exp2 in enumerate(self.experiments)
                      for s, total_fg in zip(self.requested_cls, fg_totals)}

        # Apply alm based calibration factors
        self._calibrate_spectra(cmbfg_dict, **nuis_params)

        # Introduce spectra rotations
        cmbfg_dict = self._get_rotated_spectra(cmbfg_dict, **nuis_params)

        # Introduce templates of systematics from file, if needed
        if self.systematics_template:
            cmbfg_dict = self._get_template_from_file(cmbfg_dict, **nuis_params)

        # Built theory
        dls_dict = {}
        for m in self.spec_meta:
            p = m["pol"]
            if p in ["tt", "ee", "bb"]:
                dls_dict[p, m["t1"], m["t2"]] = cmbfg_dict[p, m["t1"], m["t2"]]
            else:  # ['te','tb','eb']
                if m["hasYX_xsp"]:  # case with symmetrize = False and ET/BT/BE spectra
                    dls_dict[p, m["t2"], m["t1"]] = cmbfg_dict[p, m["t2"], m["t1"]]
                else:  # case of TE/TB/EB spectra, or symmetrize = True
                    dls_dict[p, m["t1"], m["t2"]] = cmbfg_dict[p, m["t1"], m["t2"]]

                # if symmetrize = True, dls_dict has already been set
                # equal to cmbfg_dict[p, m["t1"], m["t2"]
                # now we add cmbfg_dict[p, m["t2"], m["t1"] and we average them
                # as we do for our data
                if self.defaults["symmetrize"]:
                    dls_dict[p, m["t1"], m["t2"]] += cmbfg_dict[p, m["t2"], m["t1"]]
                    dls_dict[p, m["t1"], m["t2"]] *= 0.5

        return dls_dict

    def _get_gauss_data(self):
        """
        Get Gaussian likelihood data for use with SoLiket
        :return: GaussianData instance
        """
        from soliket.gaussian import GaussianData
        ell_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            ell_vec[m["ids"]] = m["leff"]
        return GaussianData("mflike", ell_vec, self.data_vec, self.cov, indices=self.indices_soliket)

    def _get_theory(self, **params_values):
        """
        Get theory vector (e.g. for use with SoLiket)

        :return: binned theory vector
        """
        cl = self.provider.get_Cl(ell_factor=True)
        fg_totals = self.provider.get_fg_totals()
        return self._get_power_spectra(cl, fg_totals, **params_values)

    ###########################################################################
    ## This part deals with calibration factors
    ## Here we implement an alm based calibration
    ## Each field {T,E,B}{freq1,freq2,...,freqn} gets an independent
    ## calibration factor, e.g. calT_145, calE_154, calT_225, etc..
    ## plus a calibration factor per channel, e.g. cal_145, etc...
    ## A global calibration factor calG_all is also considered.
    ###########################################################################

    def _calibrate_spectra(self, dls_dict, **nuis_params):
        r"""
        Calibrates the spectra in place through calibration factors at
        the map level:

        .. math::

           D^{{\rm cal}, TT, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}^2_{G}\, {\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm T}}\, D^{TT, \nu_1 \nu_2}_{\ell}

           D^{{\rm cal}, TE, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}^2_{G}\,{\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm E}}\, D^{TT, \nu_1 \nu_2}_{\ell}

           D^{{\rm cal}, EE, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}^2_{G}\,{\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm E}\,
           {\rm cal}^{\nu_2}_{\rm E}}\, D^{EE, \nu_1 \nu_2}_{\ell}


        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary, calibrated in place
        :param \**nuis_params: dictionary of nuisance parameters


        """
        # allowing for not having calT_{exp} in the yaml

        cal_pars = {}
        calG_all = 1 / nuis_params["calG_all"]
        if "tt" in self.requested_cls or "te" in self.requested_cls:
            cal_pars["t"] = {exp: calG_all / (nuis_params[f"cal_{exp}"] * nuis_params.get(f"calT_{exp}", 1))
                             for exp in self.experiments}

        if "ee" in self.requested_cls or "te" in self.requested_cls:
            cal_pars["e"] = {exp: calG_all / (nuis_params[f"cal_{exp}"] * nuis_params[f"calE_{exp}"])
                             for exp in self.experiments}

        self._mul_calibrations(dls_dict, cal1=cal_pars, cal2=cal_pars)

    def _mul_calibrations(self, dls_dict, cal1, cal2):
        for (spec, freq1, freq2), cl in dls_dict.items():
            if (cal := (cal1[spec[0]][freq1] * cal2[spec[1]][freq2])) != 1:
                cl *= cal

    ###########################################################################
    ## This part deals with rotation of spectra
    ## Each freq {freq1,freq2,...,freqn} gets a rotation angle alpha_93, alpha_145, etc..
    ###########################################################################

    def _get_rotated_spectra(self, dls_dict, **nuis_params):
        r"""
        Rotates the polarization spectra through polarization angles:

        .. math::

           D^{{\rm rot}, TE, \nu_1 \nu_2}_{\ell} &= \cos(\alpha^{\nu_2})
           D^{TE, \nu_1 \nu_2}_{\ell}

           D^{{\rm rot}, EE, \nu_1 \nu_2}_{\ell} &= \cos(\alpha^{\nu_1})
           \cos(\alpha^{\nu_2}) D^{EE, \nu_1 \nu_2}_{\ell}

        It uses the ``syslibrary.syslib_mflike.Rotation_alm`` function.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param \**nuis_params: dictionary of nuisance parameters

        :return: dictionary of rotated CMB+foregrounds :math:`D_{\ell}`
        """

        # allowing for not having polarization angles in the yaml

        rot_pars = [nuis_params.get(f"alpha_{exp}", 0) for exp in self.experiments]
        if not any(rot_pars):
            return dls_dict

        from syslibrary import syslib_mflike as syl

        rot = syl.Rotation_alm(ell=self.l_bpws, spectra=dls_dict)

        return rot(rot_pars, nu=self.experiments, cls=self.requested_cls)

    ###########################################################################
    ## This part deals with template marginalization
    ## A dictionary of template dls is read from yaml (likely to be not efficient)
    ## then rescaled and added to theory dls
    ###########################################################################

    # This is slow, but should be done only once
    def _init_template_from_file(self):
        """
        Reads the systematics template from file, using
        the ``syslibrary.syslib.ReadTemplateFromFile``
        function.
        """
        if not (root := self.systematics_template.get("rootname")):
            raise LoggedError(self.log, "Missing 'rootname' for systematics template!")

        from syslibrary import syslib

        # decide where to store systematics template.
        # Currently stored inside syslibrary package
        templ_from_file = syslib.ReadTemplateFromFile(rootname=root)
        self.dltempl_from_file = templ_from_file(ell=self.l_bpws)

    def _get_template_from_file(self, dls_dict, **nuis_params):
        r"""
        Adds the systematics template, modulated by ``nuis_params['templ_freq']``
        parameters, to the :math:`D_{\ell}`.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param \**nuis_params: dictionary of nuisance parameters

        :return: dictionary of CMB+foregrounds :math:`D_{\ell}`
                 with systematics templates
        """
        # templ_pars=[nuis_params['templ_'+str(exp)] for exp in self.experiments]
        # templ_pars currently hard-coded
        # but ideally should be passed as input nuisance
        templ_pars = {
            cls: np.zeros((len(self.experiments), len(self.experiments)))
            for cls in self.requested_cls
        }

        for cls in self.requested_cls:
            for i1, exp1 in enumerate(self.experiments):
                for i2, exp2 in enumerate(self.experiments):
                    dls_dict[cls, exp1, exp2] += (
                            templ_pars[cls][i1][i2] * self.dltempl_from_file[cls, exp1, exp2]
                    )

        return dls_dict


class TTTEEE(_MFLike):
    ...


class TTEE(_MFLike):
    ...


class TTTE(_MFLike):
    ...


class TEEE(_MFLike):
    ...


class TT(_MFLike):
    ...


class TE(_MFLike):
    ...


class EE(_MFLike):
    ...
