r"""
.. module:: theoryforge

The ``TheoryForge`` class applies the foreground spectra and systematics effects to the theory
spectra provided by ``MFLike``. To do that, ``TheoryForge`` gets from ``MFLike`` the appropriate
list of arrays, the requested temperature/polarization fields, the :math:`\ell` ranges, the list of
expected parameters, a dictionary of the passbands read from the ``sacc`` file:

.. code-block:: python

   bands = {"experiment_channel": {{"nu": [freqs...], "bandpass": [...]}}, ...}

This dictionary is then used to compute the bandpass transmissions, which are used for the actual
foreground spectra computation.


If one wants to use this class as standalone, the ``bands`` dictionary is filled when initializing
``TheoryForge``.

This class applies three kinds of systematic effects to the CMB + foreground power spectrum:
    * calibrations (global ``calG_all``, per channel ``cal_exp``, per field
      ``calT_exp``, ``calE_exp``)
    * polarization angles effect (``alpha_exp``)
    * systematic templates (e.g. T --> P leakage). In this case the dictionary
      ``systematics_template`` has to be filled with the correct path
      ``rootname``:

      .. code-block:: yaml

        systematics_template:
          rootname: "test_template"

If left ``null``, no systematic template is applied.

The values of the systematic parameters are set in ``MFLike.yaml``.  They have to be named as
``cal/calT/calE/alpha`` + ``_`` + experiment_channel string (e.g. ``LAT_93/dr6_pa4_f150``).


The bandpass shifts are applied within the ``_bandpass_construction`` function. There are two possibilities:
    * reading the passband :math:`\tau(\nu)` stored in a sacc file
      (which is the default now)
    * building the passbands :math:`\tau(\nu)`, either as Dirac delta or as top-hat

For the first option, it is necessary to leave the `top_hat_band` key empty in ``MFLike.yaml``:

.. code-block:: yaml

  top_hat_band: null

For the second option, the ``top_hat_band`` dictionary in ``MFLike.yaml`` has to be filled with two keys:
    * ``nsteps``: setting the number of frequencies used in the band integration
      (either 1 for a Dirac delta or > 1)
    * ``bandwidth``: setting the relative width :math:`\delta` of the band with respect to
      the central frequency, such that the frequency extremes are
      :math:`\nu_{\rm{low/high}} = \nu_{\rm{center}}(1 \mp \delta/2) + \Delta^{\nu}_{\rm band}`
      (with :math:`\Delta^{\nu}_{\rm band}` being the possible bandpass shift).
      ``bandwidth`` has to be 0 if ``nstep`` = 1, > 0 otherwise.
      ``bandwidth`` can be a list if you want a different width for each band e.g. ``bandwidth: [0.3,0.2,0.3]`` for 3 bands.

The effective frequencies, used as central frequencies to build the bandpasses, are read from the
``bands`` dictionary as before. To build a Dirac delta, use:

.. code-block:: yaml

  top_hat_band:
    nsteps: 1
    bandwidth: 0

"""

import os
from itertools import product

import numpy as np
from cobaya.log import LoggedError


# Converts from cmb temperature to differential source intensity
# (see eq. 8 of https://arxiv.org/abs/1303.5070).
# The bandpass transmission needs to be divided by
# nu^2 if measured with respect to a RJ source.
# This factor is already included here.
def _cmb2bb(nu):
    r"""
    Computes the conversion factor :math:`\frac{\partial B_{\nu}}{\partial T}`
    from CMB thermodynamic units to differential source intensity.
    Passbands measured with respect to a RJ source have to be divided by a
    :math:`\nu^2` factor.

    Numerical constants are not included, which is not a problem when using this
    conversion both at numerator and denominator.

    :param nu: frequency array

    :return: the array :math:`\frac{\partial B_{\nu}}{\partial T}`. See note above.
    """
    from scipy import constants

    T_CMB = 2.72548
    x = nu * constants.h * 1e9 / constants.k / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


class TheoryForge:
    def __init__(self, mflike=None):
        if mflike is None:
            import logging

            self.log = logging.getLogger(self.__class__.__name__.lower())
            self.data_folder = None
            self.experiments = np.array(["LAT_93", "LAT_145", "LAT_225"])
            self.foregrounds = {
                "normalisation": {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725},
                "components": {
                    "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                    "te": ["radio", "dust"],
                    "ee": ["radio", "dust"],
                },
            }
            self.l_bpws = np.arange(2, 6002)
            self.requested_cls = ["tt", "te", "ee"]
            self.bandint_freqs = np.array([93.0, 145.0, 225.0])
            self.use_top_hat_band = False
            self.bands = {
                f"{exp}_s0": {"nu": [self.bandint_freqs[iexp]], "bandpass": [1.0]}
                for iexp, exp in enumerate(self.experiments)
            }
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.experiments = mflike.experiments
            self.foregrounds = mflike.foregrounds
            self.bands = mflike.bands
            self.l_bpws = mflike.l_bpws
            self.requested_cls = mflike.requested_cls
            self.expected_params_fg = mflike.expected_params_fg
            self.expected_params_nuis = mflike.expected_params_nuis
            self.spec_meta = mflike.spec_meta
            self.defaults_cuts = mflike.defaults

            # Initialize foreground model
            self._init_foreground_model()

            # Parameters for template from file
            self.use_systematics_template = bool(mflike.systematics_template)

            if self.use_systematics_template:
                self.systematics_template = mflike.systematics_template
                # Initialize template for marginalization, if needed
                self._init_template_from_file()

            # Parameters for band integration
            self.use_top_hat_band = bool(mflike.top_hat_band)
            if self.use_top_hat_band:
                self.bandint_nsteps = mflike.top_hat_band["nsteps"]
                self.bandint_width = mflike.top_hat_band["bandwidth"]

                # checks on the bandpass input params, to be done only at the initialization
                if not hasattr(self.bandint_width, "__len__"):
                    self.bandint_width = np.full_like(
                        self.experiments, self.bandint_width, dtype=float
                    )
                if self.bandint_nsteps > 1 and np.any(np.array(self.bandint_width) == 0):
                    raise LoggedError(
                        self.log, "One band has width = 0, set a positive width and run again"
                    )

    # Takes care of the bandpass construction. It returns a list of nu-transmittance
    # for each frequency or an array with the effective freqs.
    # bandpasses saved in the sacc file have to be divided by nu^2
    # if measured with respect to a RJ source.
    # This factor is already included in the _cmb2bb function
    def _bandpass_construction(self, **params):
        r"""
        Builds the bandpass transmission
        :math:`\frac{\frac{\partial B_{\nu+\Delta \nu}}{\partial T}
        \tau(\nu+\Delta \nu)}{\int d\nu
        \frac{\partial B_{\nu+\Delta \nu}}{\partial T} \tau(\nu+\Delta \nu)}`
        using passbands :math:`\tau(\nu)` (divided by :math:`\nu^2` if
        measured with respect to a RJ source, not read from a txt
        file) and bandpass shift :math:`\Delta \nu`. As a default,
        :math:`\tau(\nu)` is read from the sacc file.
        If ``use_top_hat_band``, :math:`\tau(\nu)` is built as a top-hat
        with width ``bandint_width`` and number of samples ``nsteps``,
        read from the ``MFLike.yaml``.
        If ``nstep = 1`` and ``bandint_width = 0``, the passband is a Dirac delta
        centered at :math:`\nu+\Delta \nu`.

        :param \**params: dictionary of nuisance parameters
        :return: the list of [nu, transmission] in the multifrequency case
                 or just an array of frequencies in the single frequency one
        """
        data_are_monofreq = False
        self.bandint_freqs = []
        for iexp, exp in enumerate(self.experiments):
            bandpar = f"bandint_shift_{exp}"
            # Only temperature bandpass for the time being
            bands = self.bands[f"{exp}_s0"]
            nu_ghz, bp = np.asarray(bands["nu"]), np.asarray(bands["bandpass"])
            # computing top-hat bandpass to make band integration
            if self.use_top_hat_band:
                # Compute central frequency given bandpass in the sacc file
                fr = nu_ghz @ bp / bp.sum()
                if self.bandint_nsteps > 1:
                    bandlow = fr * (1 - self.bandint_width[iexp] * 0.5)
                    bandhigh = fr * (1 + self.bandint_width[iexp] * 0.5)
                    # nubtrue = np.linspace(bandlow, bandhigh, self.bandint_nsteps, dtype=float)
                    nub = np.linspace(
                        bandlow + params[bandpar],
                        bandhigh + params[bandpar],
                        self.bandint_nsteps,
                        dtype=float,
                    )
                    tranb = _cmb2bb(nub)
                    # normalization integral to be evaluated at the shifted freqs
                    # in order to have cmb component calibrated to 1
                    tranb_norm = np.trapz(_cmb2bb(nub), nub)
                    self.bandint_freqs.append([nub, tranb / tranb_norm])
                # in case we don't want to do band integration, e.g. when we have multifreq bandpass in sacc file
                if self.bandint_nsteps == 1:
                    nub = fr + params[bandpar]
                    data_are_monofreq = True
                    self.bandint_freqs.append(nub)
            # using the bandpass from sacc file
            else:
                nub = nu_ghz + params[bandpar]
                if len(bp) == 1:
                    # Monofrequency channel
                    data_are_monofreq = True
                    self.bandint_freqs.append(nub[0])
                else:
                    trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
                    trans = bp / trans_norm * _cmb2bb(nub)
                    self.bandint_freqs.append([nub, trans])

        # fgspectra can't mix monofrequency with [nu, bp]. If one channel is mono-frequency then we
        # assume all of them and pass to fgspectra an array (not list!!) of frequencies
        if data_are_monofreq:
            self.bandint_freqs = np.asarray(self.bandint_freqs)
            self.log.info("bandpass is delta function, no band integration performed")

    def get_modified_theory(self, Dls, **params):
        r"""
        Takes the theory :math:`D_{\ell}`, sums it to the total
        foreground power spectrum (possibly computed with bandpass
        shift and bandpass integration) computed by ``_get_foreground_model``
        and applies calibration,
        polarization angles rotation and systematic templates.

        :param Dls: CMB theory spectra
        :param \**params: dictionary of nuisance and foregrounds parameters

        :return: the CMB+foregrounds :math:`D_{\ell}` dictionary,
                 modulated by systematics
        """
        fg_params = {k: params[k] for k in self.expected_params_fg}
        nuis_params = {k: params[k] for k in self.expected_params_nuis}

        # compute bandpasses at each step only if bandint_shift params are not null
        # and bandint_freqs has been computed at least once
        if np.all(
            np.array([nuis_params[k] for k in nuis_params.keys() if "bandint_shift_" in k]) == 0.0
        ):
            if not hasattr(self, "bandint_freqs"):
                self.log.info("Computing bandpass at first step, no shifts")
                self._bandpass_construction(**nuis_params)
        else:
            self._bandpass_construction(**nuis_params)

        fg_dict = self._get_foreground_model(**fg_params)

        cmbfg_dict = {}
        # Sum CMB and FGs
        for exp1, exp2 in product(self.experiments, self.experiments):
            for s in self.requested_cls:
                cmbfg_dict[s, exp1, exp2] = Dls[s] + fg_dict[s, "all", exp1, exp2]

        # Apply alm based calibration factors
        cmbfg_dict = self._get_calibrated_spectra(cmbfg_dict, **nuis_params)

        # Introduce spectra rotations
        cmbfg_dict = self._get_rotated_spectra(cmbfg_dict, **nuis_params)

        # Introduce templates of systematics from file, if needed
        if self.use_systematics_template:
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
                if self.defaults_cuts["symmetrize"]:
                    dls_dict[p, m["t1"], m["t2"]] += cmbfg_dict[p, m["t2"], m["t1"]]
                    dls_dict[p, m["t1"], m["t2"]] *= 0.5

        return dls_dict

    ###########################################################################
    ## This part deals with foreground construction and bandpass integration ##
    ###########################################################################

    # Initializes the foreground model. It sets the SED and reads the templates
    def _init_foreground_model(self):
        """
        Initializes the foreground models from ``fgspectra``. Sets the SED
        of kSZ, tSZ, dust, radio, CIB Poisson and clustered,
        tSZxCIB, and reads the templates for CIB and tSZxCIB.
        """
        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp

        template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)), "data")
        cibc_file = os.path.join(template_path, "cl_cib_Choi2020.dat")

        # set pivot freq and multipole
        self.fg_nu_0 = self.foregrounds["normalisation"]["nu_0"]
        self.fg_ell_0 = self.foregrounds["normalisation"]["ell_0"]

        # We don't seem to be using this
        # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        self.cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
        self.cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
        self.dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.tSZ_and_CIB = fgc.SZxCIB_Choi2020()

        components = self.foregrounds["components"]
        self.fg_component_list = {s: components[s] for s in self.requested_cls}

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model(self, ell=None, freqs_order=None, **fg_params):
        r"""
        Gets the foreground power spectra for each component computed by ``fgspectra``.
        The computation assumes the bandpass transmissions computed in ``_bandpass_construction``
        and integration in frequency is performed if the passbands are not Dirac delta.

        :param ell: ell range. If ``None`` the default range
                    set in ``mflike.l_bpws`` is used
        :param freqs_order: list of the effective frequencies for each channel
                          used to compute the foreground components. Useful when
                          this function is called outside of mflike, used in place of
                          ``self.experiments``
        :param \**fg_params: parameters of the foreground components

        :return: the foreground dictionary
        """

        # if ell = None, it uses the l_bpws, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if not hasattr(ell, "__len__"):
            ell = self.l_bpws
        ell_0 = self.fg_ell_0
        nu_0 = self.fg_nu_0

        # Normalisation of radio sources
        ell_clp = ell * (ell + 1.0)
        ell_0clp = ell_0 * (ell_0 + 1.0)

        model = {}
        model["tt", "kSZ"] = fg_params["a_kSZ"] * self.ksz(
            {"nu": self.bandint_freqs}, {"ell": ell, "ell_0": ell_0}
        )
        model["tt", "cibp"] = fg_params["a_p"] * self.cibp(
            {
                "nu": self.bandint_freqs,
                "nu_0": nu_0,
                "temp": fg_params["T_d"],
                "beta": fg_params["beta_p"],
            },
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_p"]},
        )
        model["tt", "radio"] = fg_params["a_s"] * self.radio(
            {"nu": self.bandint_freqs, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["tt", "tSZ"] = fg_params["a_tSZ"] * self.tsz(
            {"nu": self.bandint_freqs, "nu_0": nu_0},
            {"ell": ell, "ell_0": ell_0},
        )
        model["tt", "cibc"] = fg_params["a_c"] * self.cibc(
            {
                "nu": self.bandint_freqs,
                "nu_0": nu_0,
                "temp": fg_params["T_d"],
                "beta": fg_params["beta_c"],
            },
            {"ell": ell, "ell_0": ell_0},
        )
        model["tt", "dust"] = fg_params["a_gtt"] * self.dust(
            {
                "nu": self.bandint_freqs,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {"ell": ell, "ell_0": 500.0, "alpha": fg_params["alpha_dT"]},
        )
        model["tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(
            {
                "kwseq": (
                    {"nu": self.bandint_freqs, "nu_0": nu_0},
                    {
                        "nu": self.bandint_freqs,
                        "nu_0": nu_0,
                        "temp": fg_params["T_d"],
                        "beta": fg_params["beta_c"],
                    },
                )
            },
            {
                "kwseq": (
                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                    {
                        "ell": ell,
                        "ell_0": ell_0,
                        "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"]),
                    },
                )
            },
        )

        model["ee", "radio"] = fg_params["a_psee"] * self.radio(
            {"nu": self.bandint_freqs, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["ee", "dust"] = fg_params["a_gee"] * self.dust(
            {
                "nu": self.bandint_freqs,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {"ell": ell, "ell_0": 500.0, "alpha": fg_params["alpha_dE"]},
        )

        model["te", "radio"] = fg_params["a_pste"] * self.radio(
            {"nu": self.bandint_freqs, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["te", "dust"] = fg_params["a_gte"] * self.dust(
            {
                "nu": self.bandint_freqs,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {"ell": ell, "ell_0": 500.0, "alpha": fg_params["alpha_dE"]},
        )

        fg_dict = {}
        if not hasattr(freqs_order, "__len__"):
            experiments = self.experiments
        else:
            experiments = freqs_order
        for c1, exp1 in enumerate(experiments):
            for c2, exp2 in enumerate(experiments):
                for s in self.requested_cls:
                    fg_dict[s, "all", exp1, exp2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if comp == "tSZ_and_CIB":
                            fg_dict[s, "tSZ", exp1, exp2] = model[s, "tSZ"][c1, c2]
                            fg_dict[s, "cibc", exp1, exp2] = model[s, "cibc"][c1, c2]
                            fg_dict[s, "tSZxCIB", exp1, exp2] = (
                                model[s, comp][c1, c2]
                                - model[s, "tSZ"][c1, c2]
                                - model[s, "cibc"][c1, c2]
                            )
                            fg_dict[s, "all", exp1, exp2] += model[s, comp][c1, c2]
                        else:
                            fg_dict[s, comp, exp1, exp2] = model[s, comp][c1, c2]
                            fg_dict[s, "all", exp1, exp2] += fg_dict[s, comp, exp1, exp2]

        return fg_dict

    ###########################################################################
    ## This part deals with calibration factors
    ## Here we implement an alm based calibration
    ## Each field {T,E,B}{freq1,freq2,...,freqn} gets an independent
    ## calibration factor, e.g. calT_145, calE_154, calT_225, etc..
    ## plus a calibration factor per channel, e.g. cal_145, etc...
    ## A global calibration factor calG_all is also considered.
    ###########################################################################

    def _get_calibrated_spectra(self, dls_dict, **nuis_params):
        r"""
        Calibrates the spectra through calibration factors at
        the map level:

        .. math::

           D^{{\rm cal}, TT, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}_{G}\, {\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm T}}\, D^{TT, \nu_1 \nu_2}_{\ell}

           D^{{\rm cal}, TE, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}_{G}\,{\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm E}}\, D^{TT, \nu_1 \nu_2}_{\ell}

           D^{{\rm cal}, EE, \nu_1 \nu_2}_{\ell} &= \frac{1}{
           {\rm cal}_{G}\,{\rm cal}^{\nu_1} \, {\rm cal}^{\nu_2}\,
           {\rm cal}^{\nu_1}_{\rm E}\,
           {\rm cal}^{\nu_2}_{\rm E}}\, D^{EE, \nu_1 \nu_2}_{\ell}

        It uses the ``syslibrary.syslib_mflike.Calibration_alm`` function.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param \**nuis_params: dictionary of nuisance parameters

        :return: dictionary of calibrated CMB+foregrounds :math:`D_{\ell}`
        """
        from syslibrary import syslib_mflike as syl

        # allowing for not having calT_{exp} in the yaml

        cal_pars = {}
        if "tt" in self.requested_cls or "te" in self.requested_cls:
            cal = nuis_params["calG_all"] * np.array(
                [
                    nuis_params[f"cal_{exp}"] * nuis_params.get(f"calT_{exp}", 1)
                    for exp in self.experiments
                ]
            )
            cal_pars["t"] = 1 / cal

        if "ee" in self.requested_cls or "te" in self.requested_cls:
            cal = nuis_params["calG_all"] * np.array(
                [nuis_params[f"cal_{exp}"] * nuis_params[f"calE_{exp}"] for exp in self.experiments]
            )
            cal_pars["e"] = 1 / cal

        calib = syl.Calibration_alm(ell=self.l_bpws, spectra=dls_dict)

        return calib(cal1=cal_pars, cal2=cal_pars, nu=self.experiments)

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
        from syslibrary import syslib_mflike as syl

        # allowing for not having polarization angles in the yaml

        rot_pars = [nuis_params.get(f"alpha_{exp}", 0) for exp in self.experiments]

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
        the ``syslibrary.syslib_mflike.ReadTemplateFromFile``
        function.
        """
        if not self.systematics_template.get("rootname"):
            raise LoggedError(self.log, "Missing 'rootname' for systematics template!")

        from syslibrary import syslib_mflike as syl

        # decide where to store systematics template.
        # Currently stored inside syslibrary package
        templ_from_file = syl.ReadTemplateFromFile(rootname=self.systematics_template["rootname"])
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
