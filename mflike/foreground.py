r"""
.. module:: foreground

The ``Foreground`` class calculates foreground spectra , using :math:`\ell` ranges, array of frequencies,etc.
The inherited ``BandpowerForeground`` adds integration over bandpowers, using the bandpass transmissions.

If one wants to use this class as standalone, the ``bands`` dictionary is filled when initializing
``BandpowerForeground``.

The values of the systematic parameters are set in ``MFLike.yaml``.  They have to be named as
``cal/calT/calE/alpha`` + ``_`` + experiment_channel string (e.g. ``LAT_93/dr6_pa4_f150``).


The bandpass shifts are applied within the ``_bandpass_construction`` function. There are two possibilities:
    * reading the passband :math:`\tau(\nu)` stored in a sacc file
      (which is the default now)
    * building the passbands :math:`\tau(\nu)`, either as Dirac delta or as top-hat

For the second option, the ``top_hat_band`` dictionary in ``MFLike.yaml`` has to be filled with two keys:
    * ``nsteps``: setting the number of frequencies used in the band integration
      (either 1 for a Dirac delta or > 1)
    * ``bandwidth``: setting the relative width :math:`\delta` of the band with respect to
      the central frequency, such that the frequency extremes are
      :math:`\nu_{\rm{low/high}} = \nu_{\rm{center}}(1 \mp \delta/2) + \Delta^{\nu}_{\rm band}`
      (with :math:`\Delta^{\nu}_{\rm band}` being the possible bandpass shift).
      ``bandwidth`` has to be 0 if ``nstep`` = 1, > 0 otherwise.
      ``bandwidth`` can be a list if you want a different width for each band
      e.g. ``bandwidth: [0.3,0.2,0.3]`` for 3 bands.

The effective frequencies, used as central frequencies to build the bandpasses, are read from the
``bands`` dictionary as before. To build a Dirac delta, use:

.. code-block:: yaml

  top_hat_band:
    nsteps: 1
    bandwidth: 0

"""

import os

import numpy as np
from cobaya.log import LoggedError
from cobaya.theory import Provider, Theory
from scipy import constants

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

nuis_params_defaults = {
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.20,
    "a_c": 4.90,
    "beta_c": 2.20,
    "a_s": 3.10,
    "a_gtt": 2.80,
    "a_gte": 0.10,
    "a_gee": 0.10,
    "a_psee": 0.0,
    "a_pste": 0.0,
    "xi": 0.10,
    "T_d": 9.60,
    "beta_s": -2.5,
    "alpha_s": 1.0,
    "T_effd": 19.6,
    "beta_d": 1.5,
    "alpha_dT": -0.6,
    "alpha_dE": -0.4,
    "alpha_p": 1.0,
    "alpha_tSZ": 0.0
}

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
    T_CMB = 2.72548
    x = nu * constants.h * 1e9 / constants.k / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


class Foreground(Theory):
    normalisation: dict
    components: dict
    experiments: list[str]
    lmin: int
    lmax: int
    requested_cls: list[str]
    bandint_freqs: list
    ells: np.ndarray

    # Initializes the foreground model. It sets the SED and reads the templates
    def initialize(self):
        """
        Initializes the foreground models from ``fgspectra``. Sets the SED
        of kSZ, tSZ, dust, radio, CIB Poisson and clustered,
        tSZxCIB, and reads the templates for CIB and tSZxCIB.
        """
        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp

        template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)), "data")
        tsz_file = os.path.join(template_path, "cl_tsz_150_bat.dat")
        cibc_file = os.path.join(template_path, "cl_cib_Choi2020.dat")
        cibxtsz_file = os.path.join(template_path, "cl_sz_x_cib.dat")

        # set pivot freq and multipole
        self.fg_nu_0 = self.normalisation["nu_0"]
        self.fg_ell_0 = self.normalisation["ell_0"]

        # We don't seem to be using this
        # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        self.cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.PowerLawRescaledTemplate(tsz_file))
        self.cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
        self.dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())

        tsz_cib_sed = fgf.Join(fgf.ThermalSZ(), fgf.CIB())
        tsz_cib_power_spectra = [
            fgp.PowerLawRescaledTemplate(tsz_file),
            fgp.PowerSpectrumFromFile(cibc_file),
            fgp.PowerSpectrumFromFile(cibxtsz_file)
        ]
        tsz_cib_cl = fgp.PowerSpectraAndCovariance(*tsz_cib_power_spectra)

        self.tSZ_and_CIB = fgc.CorrelatedFactorizedCrossSpectrum(tsz_cib_sed, tsz_cib_cl)

        if self.ells is None:
            self.ells = np.arange(self.lmin, self.lmax + 1)

    def initialize_with_provider(self, provider: Provider):
        self.fg_component_list = {s: self.components[s] for s in self.requested_cls}

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model_arrays(self, fg_params, ell=None):
        r"""
        Gets the foreground power spectra for each component computed by ``fgspectra``.
        Integration over frequency is performed using bandint_freqs.

        :param fg_params: parameters of the foreground components
        :param ell: ell range. If ``None`` the default range
            set in ``mflike.l_bpws`` is used

        :return: the foreground dictionary of arrays
        """

        # if ell = None, it uses the l_bpws, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if ell is None:
            ell = self.ells
        ell_0 = self.fg_ell_0
        nu_0 = self.fg_nu_0

        # Normalisation of radio sources
        ell_clp = ell * (ell + 1.0)
        ell_0clp = ell_0 * (ell_0 + 1.0)

        model = {}
        if "tt" in self.requested_cls:
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
                {"ell": ell, "ell_0": ell_0, "alpha": fg_params["alpha_tSZ"]},
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
                        {
                            "ell": ell,
                            "ell_0": ell_0,
                            "amp": fg_params["a_tSZ"],
                            "alpha": fg_params["alpha_tSZ"]
                        },
                        {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                        {
                            "ell": ell,
                            "ell_0": ell_0,
                            "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"]),
                        },
                    )
                },
            )

        if "ee" in self.requested_cls:
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

        if "te" in self.requested_cls:
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

        return model

    def get_foreground_model(self, ell=None, freqs_order=None, **fg_params):
        r"""
        Gets the foreground power spectra for each component computed by ``fgspectra``.
        Integration over frequency is performed using bandint_freqs.
        This function is not used by Cobaya, but can be used to get the individual
        foreground components and total as a dictionary when accessing the class
        separately.

        :param ell: ell range. If ``None`` the default range
                    set in ``l_bpws`` is used
        :param freqs_order: list of the effective frequencies for each channel
                          used to compute the foreground components. Useful when
                          this class is called outside of mflike, used in place of
                          ``self.experiments``
        :param **fg_params: parameters of the foreground components

        :return: the foreground dictionary
        """

        if ell is None:
            ell = self.ells
        model = self._get_foreground_model_arrays(fg_params, ell=ell)
        experiments = self.experiments if freqs_order is None else freqs_order
        fg_dict = {}
        for c1, exp1 in enumerate(experiments):
            for c2, exp2 in enumerate(experiments):
                for s in self.requested_cls:
                    sum_all = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        term = model[s, comp][c1, c2]
                        if comp == "tSZ_and_CIB":
                            fg_dict[s, "tSZ", exp1, exp2] = model[s, "tSZ"][c1, c2]
                            fg_dict[s, "cibc", exp1, exp2] = model[s, "cibc"][c1, c2]
                            fg_dict[s, "tSZxCIB", exp1, exp2] = (
                                    term - model[s, "tSZ"][c1, c2] - model[s, "cibc"][c1, c2]
                            )
                        else:
                            fg_dict[s, comp, exp1, exp2] = term
                        sum_all += term
                    fg_dict[s, "all", exp1, exp2] = sum_all
        return fg_dict

    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Fills the ``state`` dictionary of the ``Foreground`` Theory class
        with the foreground spectra, computed using the bandpass
        transmissions the sampled foreground  parameters.

        :param state: ``state`` dictionary to be filled with computed foreground
                      spectra
        :param want_derived: if derived wanted (none here)
        :param **params_values_dict: dictionary of parameters from the sampler
        """

        state["fg_totals"] = self.get_foreground_model_totals(**params_values_dict)

    def get_foreground_model_totals(self, requested_cl=(), **params_values_dict):
        """
        Get total foregrounds for each cl type and frequency channel.

        :param requested_cl: optional list of cl types to compute (tt, ee, te)
        :param params_values_dict: foreground parameters
        :return: list of arrays for each requested_cl
        """
        # get total foregrounds; model is dictionary of arrays for each frequency combo
        model = self._get_foreground_model_arrays(params_values_dict)
        return [np.sum([model[s, comp] for comp in self.fg_component_list[s]], axis=0)
                for s in (requested_cl if requested_cl else self.requested_cls)]

    def get_fg_totals(self):
        """
        Returns the ``state`` dictionary of foreground spectra, when used with Cobaya.
        Should only be called after the model is calculated by Cobaya.
        """
        return self.current_state["fg_totals"]

    def must_provide(self, **requirements):
        if (req := requirements.get("fg_totals")) is not None:
            self.requested_cls = req.get("requested_cls", self.requested_cls)
            self.ells = req.get("ells", self.ells)
            self.experiments = req.get("experiments", self.experiments)


class BandpowerForeground(Foreground):
    # foregrounds integrated over bandpass windows

    top_hat_band: dict = None
    bands: dict = None

    def initialize(self):
        super().initialize()
        super().initialize_with_provider(self)
        if self.bands is None:
            self.bands = {
                f"{exp}_s0": {"nu": [self.bandint_freqs[iexp]], "bandpass": [1.0]}
                for iexp, exp in enumerate(self.experiments)}
        self._initialized = False
        self.init_bandpowers()

    def init_bandpowers(self):
        self.use_top_hat_band = bool(self.top_hat_band)
        # Parameters for band integration
        if self.use_top_hat_band:
            self.bandint_nsteps = self.top_hat_band["nsteps"]
            self.bandint_width = self.top_hat_band["bandwidth"]

            # checks on the bandpass input params, to be done only at the initialization
            if not hasattr(self.bandint_width, "__len__"):
                self.bandint_width = np.full_like(
                    self.experiments, self.bandint_width, dtype=float
                )
            if self.bandint_nsteps > 1 and np.any(np.array(self.bandint_width) == 0):
                raise LoggedError(
                    self.log, "One band has width = 0, set a positive width and run again"
                )
        self._bandint_shift_params = [f"bandint_shift_{f}" for f in self.experiments]
        # default bandpass when shift is 0
        shift_params = dict.fromkeys(self._bandint_shift_params, 0.0)
        self._bandpass_construction(**shift_params)

    def must_provide(self, **requirements):
        # fg_dict is required by mflike
        # and requires some params to be computed
        # Assign those as requested or us defaults
        # otherwise use default values
        # Foreground requires bandint_freqs from BandPass
        # TODO: not clear that changing number of freqs actually works (for shift parameters)
        super().must_provide(**requirements)
        if (req := requirements.get("fg_totals")) is not None:
            self.bands = req.get("bands", self.bands)
            self.top_hat_band = req.get("top_hat_band", self.top_hat_band)
            self.init_bandpowers()

    def get_can_support_params(self):
        return self._bandint_shift_params

    def _get_foreground_model_arrays(self, fg_params, ell=None):
        r"""
        Gets the foreground power spectra for each component computed by ``fgspectra``.
        The computation assumes the bandpass transmissions computed in ``_bandpass_construction``
        and integration in frequency is performed if the passbands are not Dirac delta.

        :param fg_params: parameters of the foreground components
        :param ell: ell range. If ``None`` the default range
            set in ``self.ells`` is used

        :return: the foreground dictionary of arrays
        """

        # compute bandpasses at each step only if bandint_shift params are not null
        # and bandint_freqs has been computed at least once
        if any(fg_params.get(k) for k in self._bandint_shift_params):
            self._bandpass_construction(**fg_params)
        return super()._get_foreground_model_arrays(fg_params, ell=ell)

    # Takes care of the bandpass construction. It returns a list of nu-transmittance
    # for each frequency or an array with the effective freqs.
    # bandpasses saved in the sacc file have to be divided by nu^2
    # if measured with respect to a RJ source.
    # This factor is already included in the _cmb2bb function
    def _bandpass_construction(self, _initialize=False, **params):
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
        for iexp, (band_shift, exp) in enumerate(zip(self._bandint_shift_params, self.experiments)):
            # Only temperature bandpass for the time being
            bands = self.bands[f"{exp}_s0"]
            shift = params.get(band_shift, 0.0)
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
                        bandlow + shift,
                        bandhigh + shift,
                        self.bandint_nsteps,
                        dtype=float,
                    )
                    tranb = _cmb2bb(nub)
                    # normalization integral to be evaluated at the shifted freqs
                    # in order to have cmb component calibrated to 1
                    tranb_norm = trapezoid(_cmb2bb(nub), nub)
                    self.bandint_freqs.append([nub, tranb / tranb_norm])
                # in case we don't want to do band integration, e.g. when we have multifreq bandpass in sacc file
                if self.bandint_nsteps == 1:
                    nub = fr + shift
                    data_are_monofreq = True
                    self.bandint_freqs.append(nub)
            # using the bandpass from sacc file
            else:
                nub = nu_ghz + shift
                if len(bp) == 1:
                    # Monofrequency channel
                    data_are_monofreq = True
                    self.bandint_freqs.append(nub[0])
                else:
                    trans_norm = trapezoid(bp * _cmb2bb(nub), nub)
                    trans = bp / trans_norm * _cmb2bb(nub)
                    self.bandint_freqs.append([nub, trans])

        # fgspectra can't mix monofrequency with [nu, bp]. If one channel is mono-frequency then we
        # assume all of them and pass to fgspectra an array (not list!!) of frequencies
        if data_are_monofreq:
            self.bandint_freqs = np.asarray(self.bandint_freqs)
            if self._initialized:
                self.log.info("bandpass is delta function, no band integration performed")
        self._initialized = True
