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

The values of the systematic parameters are set in ``MFLike.yaml``.  They have to be named as
``cal/calT/calE/alpha`` + ``_`` + experiment_channel string (e.g. ``LAT_93/dr6_pa4_f150``).


The bandpass shifts are applied within the ``_bandpass_construction`` function. There are two possibilities:
    * reading the passband :math:`\tau(\nu)` stored in a sacc file
      (which is the default now)
    * building the passbands :math:`\tau(\nu)`, either as Dirac delta or as top-hat

For the first option, it is necessary to leave the ``top_hat_band`` key empty in ``MFLike.yaml``:

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

If we want to neglect the beam chromaticity effect, we should leave the ``beam_profile`` key empty
in ``MFLike.yaml``:

.. code-block:: yaml

  beam_profile: null

If we want to consider it, we have several options on how to compute/read the beam profiles. Notice that we need arrays(freqs, ells+2) (computed from :math:`\ell = 0`), since we want a beam window function for each freq in the bandpasses. We should use this block in ``MFLike.yaml``:

.. code-block:: yaml

  beam_profile:
    Gaussian_beam: dict/False/null
    beam_from_file: "filename"/False/null
    
There are several options: 
    * reading the beams from the sacc file (``Gaussian_beam: False/null``, ``beam_from_file: False/null``). 
      The beams have to be stored in the ``sacc.tracers[exp].beam`` tracer 
      (this is not working so far, since the sacc beam tracer doesn't like an array(freq, ell))
    * reading the beams from an external yaml file (``Gaussian_beam: False/null``, ``beam_from_file: "filename"``). 
      Do not use the ".yaml" extension nor the path to the file, which has to be the same as the 
      data path. The yaml file has to be a dictionary ``{"{exp}_s0": {"nu": nu, 
      "beams": array(freqs, ells+2)}, "{exp}_s2": {"nu": nu, "beams": array(freqs, ells+2)},...}``
    * computing the beams as Gaussian beams (``Gaussian_beam: dict``, ``beam_from_file: ...``). When 
      ``Gaussian_beam`` is not empty, the beam is automatically computed within the code. Both T and 
      polarization Gaussian beams are computed through ``healpy.gauss_beam``. We need to pass a
      dictionary with the ``FWHM_0``, ``nu_0``, ``alpha`` parameters for each array/experiment (both in T and pol),
      in order to parametrize the Gaussian FWHM as :math:`FWHM(\nu) = FWHM(\nu_0) \left( \frac{\nu}{\nu_0} \right)^{-\alpha/2}`:

.. code-block:: yaml

  beam_profile:
    Gaussian_beam: 
      LAT_93_s0:
        FWHM_0: ...
        nu_0: ...
        alpha: ...
      LAT_93_s2:
        FWHM_0: ...
        nu_0: ...
        alpha: ...
      LAT_145_s0:
        FWHM_0: ...
        nu_0: ...
        alpha: ...
      ...
    beam_from_file: null

Once computed/read, the beam profiles are saved in 

.. code-block:: python

   self.beams = {"{exp}_s0": {"nu": nu, "beams": array(freqs, ells+2)}, "{exp}_s2": {"nu": nu, "beams": array(freqs, ells+2)},...}. 

The beams are appropriately normalized, then we select the bandpowers used in the rest of the code.

In case of bandpass shift, the chromatic beams are derived as: :math:`b^{T/P}_{\ell}(\nu + \Delta \nu) =  b^{T/P}_{\ell (\nu / \nu_0)^{-\alpha / 2}}(\nu_0 + \Delta \nu)`, starting from a monochromatic beam :math:`b^{T/P}_{\ell}(\nu_0 + \Delta \nu)`. This monochromatic beam is derived from measurements of the planet beam and assuming a certain bandpass shift :math:`\Delta \nu`. So we need a dictionary of these :math:`b^{T/P}_{\ell}(\nu_0 + \Delta \nu)` for the several values of :math:`\Delta \nu` that could be sampled in the MCMC. To apply the scaling :math:`b^{T/P}_{\ell (\nu / \nu_0)^{-\alpha / 2}}(\nu_0 + \Delta \nu)` we also need :math:`\nu_0` and :math:`\alpha` for each experiment/array. 
The array of frequencies :math:`\nu` for each experiment/array is derived from the corresponding bandpass file. 

This means that, when bandpass shifts are different from 0, we need to provide a yaml file under the key ``Bandpass_shifted_beams``:
    
.. code-block:: yaml

  beam_profile:
    Bandpass_shifted_beams: "bandpass_shifted_beams"
    Gaussian_beam: dict/False/null
    beam_from_file: "filename"/False/null

where the "bandpass_shifted_beams.yaml" file is structured as:

.. code-block:: yaml

    LAT_93_s0:
      beams: {..., '-2.0': b_ell(nu_0 -2),
                     '-1.0': b_ell(nu_0 -1),
                     ...
                     '5.0': b_ell(nu_0 + 5),
                ...}
      nu_0: ...
      alpha: ...
    LAT_93_s2:
      beams: {'-10.0': b_ell(nu_0 - 10), ...}
      nu_0: ...
      alpha: ...
    LAT_145_s0:
      beams: ...
      nu_0: ...
      alpha: ...
    ...

The "bandpass_shifted_beams.yaml" file has to be saved in the same path as the data. 

It is important the keys of ``beam_profile["Bandpass_shifted_beams"]["{exp}_s0/2"]["beams"]`` are strings of floats representing the value of :math:`\Delta \nu` (if they are strings of int the code to read the associated beams would not work).
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
            self.bandint_freqs_T = np.array([93.0, 145.0, 225.0])
            self.bandint_freqs_P = np.array([93.0, 145.0, 225.0])
            self.use_top_hat_band = False
            self.bands = {
                f"{exp}_s0": {"nu": [self.bandint_freqs_T[iexp]], "bandpass": [1.0]}
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

            self.use_beam_profile = bool(mflike.beam_profile)
            if self.use_beam_profile:
                self.beam_profile = mflike.beam_profile
                if not self.beam_profile.get("Gaussian_beam"):
                    self.beam_file = self.beam_profile.get("beam_from_file")
                    self._init_beam_from_file()
                else:
                    self.gaussian_params = self.beam_profile.get("Gaussian_beam")
                    self._init_gauss_beams()
                # reading the possible dictionary with beam profiles associated to bandpass shifts
                # this has to be present in case bandpass shifts != 0
                self.bandsh_beams_path = self.beam_profile.get("Bandpass_shifted_beams")
                if self.bandsh_beams_path:
                    print(self.bandsh_beams_path)
                    self.bandpass_shifted_beams = self._read_yaml_file(self.bandsh_beams_path)

    # Takes care of the bandpass construction. It returns a list of nu-transmittance
    # for each frequency or an array with the effective freqs.
    # bandpasses saved in the sacc file have to be divided by nu^2
    # if measured with respect to a RJ source.
    # This factor is already included in the _cmb2bb function
    def _bandpass_construction(self, **params):
        r"""
        Builds the bandpass transmission with or without beam.
        When chromatic beam is not considered, we compute:
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

        When the chromatic beam is considered, we compute
        :math:`r_{\ell}^T(\nu+\Delta \nu) = \frac{\frac{\partial B_{\nu+\Delta \nu}}{\partial T}
        \tau(\nu+\Delta \nu) b^T_{\ell}(\nu + \Delta \nu)}
        {\int d\nu
        \frac{\partial B_{\nu+\Delta \nu}}{\partial T} \tau(\nu+\Delta \nu)
        b^T_{\ell}(\nu + \Delta \nu)}`
        for the temperature field, and a corresponding expression for the polarization field,
        replacing the temperature beam with the polarization one
        :math:`b^P_{\ell}(\nu + \Delta \nu)`.

        :param \**params: dictionary of nuisance parameters
        :return: the list of [nu, transmission] in the multifrequency case
                 or just an array of frequencies in the single frequency one.
                 We distinguish between T and pol transmission when a chromatic
                 beam is included
        """
        data_are_monofreq = False
        self.bandint_freqs_T = []
        self.bandint_freqs_P = []
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

                    if not self.use_beam_profile:
                        # normalization integral to be evaluated at the shifted freqs
                        # in order to have cmb component calibrated to 1
                        tranb = _cmb2bb(nub)
                        tranb_norm = np.trapz(_cmb2bb(nub), nub)
                        self.bandint_freqs_T.append([nub, tranb / tranb_norm])
                        self.bandint_freqs_P.append([nub, tranb / tranb_norm])
                    else:
                        # fixing beams to case with Delta^nu = 0
                        blT, blP = self.return_beams(exp, nu_ghz, 0)  
                       # blT, blP = self.return_beams(exp, nu_ghz, params[bandpar])

                        tranb_normT = np.trapz(_cmb2bb(nub)[..., np.newaxis] * blT, nub, axis=0)
                        ratioT = _cmb2bb(nub)[..., np.newaxis] * blT / tranb_normT
                        self.bandint_freqs_T.append([nub, ratioT])

                        tranb_normP = np.trapz(_cmb2bb(nub)[..., np.newaxis] * blP, nub, axis=0)
                        ratioP = _cmb2bb(nub)[..., np.newaxis] * blP / tranb_normP
                        self.bandint_freqs_P.append([nub, ratioP])

                # in case we don't want to do band integration
                if self.bandint_nsteps == 1:
                    nub = fr + params[bandpar]
                    data_are_monofreq = True
                    self.bandint_freqs_T.append(nub)
                    self.bandint_freqs_P.append(nub)
            # using the bandpass from sacc file
            else:
                nub = nu_ghz + params[bandpar]
                if len(bp) == 1:
                    # Monofrequency channel
                    data_are_monofreq = True
                    self.bandint_freqs_T.append(nub[0])
                    self.bandint_freqs_P.append(nub[0])
                else:
                    if not self.use_beam_profile:
                        trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
                        trans = bp / trans_norm * _cmb2bb(nub)
                        self.bandint_freqs_T.append([nub, trans])
                        self.bandint_freqs_P.append([nub, trans])
                    else:
                        # fixing beams to case with Delta^nu = 0
                        blT, blP = self.return_beams(exp, nu_ghz, 0)
                        #blT, blP = self.return_beams(exp, nu_ghz, params[bandpar])

                        trans_normT = np.trapz(
                            bp[..., np.newaxis] * _cmb2bb(nub)[..., np.newaxis] * blT, nub, axis=0
                        )
                        ratioT = (
                            bp[..., np.newaxis] * _cmb2bb(nub)[..., np.newaxis] * blT / trans_normT
                        )
                        self.bandint_freqs_T.append([nub, ratioT])

                        trans_normP = np.trapz(
                            bp[..., np.newaxis] *  _cmb2bb(nub)[..., np.newaxis] * blP, nub, axis=0
                        )
                        ratioP = (
                            bp[..., np.newaxis] * _cmb2bb(nub)[..., np.newaxis] * blP / trans_normP
                        )
                        self.bandint_freqs_P.append([nub, ratioP])

        # fgspectra can't mix monofrequency with [nu, bp]. If one channel is mono-frequency then we
        # assume all of them and pass to fgspectra an array (not list!!) of frequencies
        if data_are_monofreq:
            self.bandint_freqs_T = np.asarray(self.bandint_freqs_T)
            self.bandint_freqs_P = np.asarray(self.bandint_freqs_P)
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
        # filling all exp x exp combinations
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
        tsz_file = os.path.join(template_path, "cl_tsz_150_bat.dat")
        cibc_file = os.path.join(template_path, "cl_cib_Choi2020.dat")
        cibxtsz_file = os.path.join(template_path, "cl_sz_x_cib.dat")

        # set pivot freq and multipole
        self.fg_nu_0 = self.foregrounds["normalisation"]["nu_0"]
        self.fg_ell_0 = self.foregrounds["normalisation"]["ell_0"]

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
        self.radioTE = fgc.FactorizedCrossSpectrumTE(fgf.PowerLaw(), fgf.PowerLaw(), fgp.PowerLaw())
        self.dustTE = fgc.FactorizedCrossSpectrumTE(
            fgf.ModifiedBlackBody(), fgf.ModifiedBlackBody(), fgp.PowerLaw()
        )

        
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
            {"nu": self.bandint_freqs_T}, {"ell": ell, "ell_0": ell_0}
        )
        model["tt", "cibp"] = fg_params["a_p"] * self.cibp(
            {
                "nu": self.bandint_freqs_T,
                "nu_0": nu_0,
                "temp": fg_params["T_d"],
                "beta": fg_params["beta_p"],
            },
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_p"]},
        )
        model["tt", "radio"] = fg_params["a_s"] * self.radio(
            {"nu": self.bandint_freqs_T, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["tt", "tSZ"] = fg_params["a_tSZ"] * self.tsz(
            {"nu": self.bandint_freqs_T, "nu_0": nu_0},
            {"ell": ell, "ell_0": ell_0, "alpha": fg_params["alpha_tSZ"]},
        )
        model["tt", "cibc"] = fg_params["a_c"] * self.cibc(
            {
                "nu": self.bandint_freqs_T,
                "nu_0": nu_0,
                "temp": fg_params["T_d"],
                "beta": fg_params["beta_c"],
            },
            {"ell": ell, "ell_0": ell_0},
        )
        model["tt", "dust"] = fg_params["a_gtt"] * self.dust(
            {
                "nu": self.bandint_freqs_T,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {"ell": ell, "ell_0": 500.0, "alpha": fg_params["alpha_dT"]},
        )
        model["tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(
            {
                "kwseq": (
                    {"nu": self.bandint_freqs_T, "nu_0": nu_0},
                    {
                        "nu": self.bandint_freqs_T,
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

        model["ee", "radio"] = fg_params["a_psee"] * self.radio(
            {"nu": self.bandint_freqs_P, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["ee", "dust"] = fg_params["a_gee"] * self.dust(
            {
                "nu": self.bandint_freqs_P,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {"ell": ell, "ell_0": 500.0, "alpha": fg_params["alpha_dE"]},
        )

        model["te", "radio"] = fg_params["a_pste"] * self.radioTE(
            {"nu": self.bandint_freqs_T, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"nu": self.bandint_freqs_P, "nu_0": nu_0, "beta": fg_params["beta_s"]},
            {"ell": ell_clp, "ell_0": ell_0clp, "alpha": fg_params["alpha_s"]},
        )
        model["te", "dust"] = fg_params["a_gte"] * self.dustTE(
            {
                "nu": self.bandint_freqs_T,
                "nu_0": nu_0,
                "temp": fg_params["T_effd"],
                "beta": fg_params["beta_d"],
            },
            {
                "nu": self.bandint_freqs_P,
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

    ###########################################################################
    ## This part deals with beam functions, i.e. reading beam from file or
    ## computing it as a Gaussian beam. We also have a function to compute
    ## the correction expected for a Gaussian beam in case of bandpass shift
    ## that should be applied to any beam profile
    ###########################################################################
    def _read_yaml_file(self, file_path):
        import yaml

        data_path = self.data_folder
        filename = os.path.join(data_path, "%s.yaml" % file_path)
        if not os.path.exists(filename):
            raise ValueError("File " + filename + " does not exist!")

        with open(filename, "r") as f:
            file = yaml.load(f, Loader=yaml.Loader)

        return file

    def _init_beam_from_file(self):
        """
        Reads the beam profile from an external file or the sacc file.
        It has to be a dictionary ``{"{exp}_s0": {"nu": nu, "beams": array(freqs, ells+2)},
        "{exp}_s2": {"nu": nu, "beams": array(freqs, ells+2)},...}``
        including temperature and polarization beams.
        """

        if not self.beam_file:
            # option to read beam from sacc.
            try:
                bool(mflike.beams)
            except:
                raise ValueError("Beams not stored in sacc files!")
            else:
                self.beams = mflike.beams
        else:
            self.beams = self._read_yaml_file(self.beam_file)

        #checking that the freq array is compatible with the bandpass one
        for exp in self.experiments:
            # checking nu is the same as the bandpass one
            if not np.allclose(self.beams[f"{exp}_s0"]['nu'], self.bands[f"{exp}_s0"]['nu'], atol = 1e-5):
                raise LoggedError(self.log, f"Frequency array for beam {exp}_s0 is not the same as the bandpass one!")
            if not np.allclose(self.beams[f"{exp}_s2"]['nu'], self.bands[f"{exp}_s2"]['nu'], atol = 1e-5):
                raise LoggedError(self.log, f"Frequency array for beam {exp}_s2 is not the same as the bandpass one!")


    def _init_gauss_beams(self):
        """
        Computes the dictionary of beams for each frequency of self.experiments
        """
        self.beams = {}
        for iexp, exp in enumerate(self.experiments):
            gauss_pars = self.gaussian_params[f"{exp}_s0"]
            FWHM0 = np.asarray(gauss_pars["FWHM_0"])
            #using the same freq array as the bandpass one
            nu = np.asarray(self.bands[f"{exp}_s0"]['nu'])
            nu0 = np.asarray(gauss_pars["nu_0"])
            alpha = np.asarray(gauss_pars["alpha"])
            # computing temperature beam for exp
            self.beams[f"{exp}_s0"] = {"nu": nu, "beams": self.gauss_beams(FWHM0, nu, nu0, alpha, False)}
            # doing the same for polarization
            gauss_pars = self.gaussian_params[f"{exp}_s2"]
            FWHM0 = np.asarray(gauss_pars["FWHM_0"])
            # nu pol should be the same as the T one, I'll comment it for now
            #nu = np.asarray(self.bands[f"{exp}_s2"]['nu'])
            nu0 = np.asarray(gauss_pars["nu_0"])
            alpha = np.asarray(gauss_pars["alpha"])
            # checking nu is the same as the bandpass one
            self.beams[f"{exp}_s2"] = {"nu": nu, "beams": self.gauss_beams(FWHM0, nu, nu0, alpha, True)}


    def gauss_beams(self, fwhm0, nu, nu0, alpha, pol):
        r"""
        Computes the Gaussian beam (either for T or pol) for each frequency of a
        frequency array according to eqs. 54/55 of arXiv:astro-ph/0008228. We assume a more general
        scaling for the FWHM: :math:`FWHM(\nu) = FWHM(\nu_0) \left( \frac{\nu}{\nu_0} \right)^{-\alpha}`.

        :param fwhm0: the FWHM for the pivot frequency
        :param nu: the frequency array in GHz
        :param nu0: the pivot frequency in GHz
        :param alpha: the exponent of the frequency scaling 
                      :math:`\left( \frac{\nu}{\nu_0} \right)^{-\alpha/2}`
        :param pol: (Bool) False to compute temperature Gaussian beam,
                    True for the polarization one

        :return: a :math:`b^{Gauss.}_{\ell}(\nu)` = ``array(freqs, lmax +2)`` with Gaussian beam
                 profiles for each frequency in :math:`\nu` (from :math:`\ell = 0`)
        """
        from astropy import constants, units
        import healpy as hp

        fwhm = fwhm0 * (nu / nu0)**(-alpha/2.)
        bls = np.empty((len(nu), self.l_bpws[-1] + 1))
        for ifw, fw in enumerate(fwhm):
            # saving the beam from ell = 2 to ell max of l_bpws
            if not pol:
                bls[ifw, :] = hp.gauss_beam(fw, lmax=self.l_bpws[-1])
            else:
                # selecting the polarized gaussian beam
                bls[ifw, :] = hp.gauss_beam(fw, lmax=self.l_bpws[-1], pol=True)[:, 1]

        return bls


    def beam_interpolation(self, b_ell_template, f_ell, ells, freqs, freq_ref, alpha):
        r'''
        Computing :math:`b_{\ell}(\nu)` from monochromatic beam :math:`b_{\ell}` using the 
        frequency scaling: :math:`(b \cdot f)_{\ell \cdot (\nu / \nu_0)^{-\alpha / 2}}`
    
        :param b_ell_template: (nell array) Template for :math:`b_{\ell}`, should be 1 at ell=0.
        :param f_ell: (nell array) Multiplicate correction to the :math:`b_{\ell}` template. 
                      Should be 1 at ell=0.
        :param ells: (nell array) ell array
        :param freqs: (nfreq array) Frequency for that experiment/array
        :param freq_ref: (float) Reference frequency.
        :param alpha: (float) Power law index.
    
        
        :return: a (nfreq, nell) array: :math:`b_{\ell}(\nu)` at each input frequency.
        '''
        from scipy.interpolate import interp1d

        #f_ell = np.ones_like(b_ell_template)
        fi = interp1d(ells, b_ell_template * f_ell, kind='linear', fill_value='extrapolate')
        bnu = fi(ells[:,np.newaxis] * (freqs / freq_ref) ** (-alpha / 2))
        # Because we extrapolate beyond lmax, output can become negative, that is 
        # unphysical so we set these to zero.
        bnu[bnu < 0] = 0
        # transposing to have an object (nfreq, nell)
        return bnu.T

    def return_beams(self, exp, nu, dnu):
        r"""
        Returns the temperature and polarization beams, properly normalized and from
        :math:`\ell = 2` (same ell range as ``self.l_bpws``). We compute them from :math:`\ell = 0`
        to normalize them in the correct way (temperature beam = 1 for :math:`\ell = 0`).
        The polarization beam is normalized by the temperature one (as in ``hp.gauss_beam``).

        In the presence of bandpass shift, we have to select the monochromatic beam :math:`b_{\ell}`
        computed from the planet beam assuming that bandpass shift. This has to be present in the 
        ``self.bandpass_shifted_beams`` dictionary. From each of these :math:`b_{\ell}`, the 
        chromatic beam is computed with the scaling :math:`b_{\ell (\nu / \nu_0)^{-\alpha / 2}}`,
        where :math:`\nu_0` and :math:`\alpha` are also found in the same dictionary.

        :param nu: the frequency array in GHz (for now, the math:`\nu` array is the same
                   between bandpass file and beam file for the same experiment/array.
                   It is passed from the ``_bandpass_construction`` function
                   for consistency.)
        :param dnu: the bandpass shift :math:`\Delta \nu`

        :return: The temperature and polarization chromatic beams
        """
        if dnu != 0:
            bandsh_beams = self.bandpass_shifted_beams[f"{exp}_s0"]
            #reading the Delta nu list in the file
            dnulist = np.array([float(dn) for dn in bandsh_beams["beams"].keys()])
            #finding the Delta nu closer to the sampled one
            Dnu = dnulist[abs(dnulist - dnu).argmin()]
            #reading the corresponding monochromatic beam 
            #the dnu keys have to be strings of floats, not int
            b = bandsh_beams["beams"][f"{Dnu}"]
            nu = np.asarray(self.bands[f"{exp}_s0"]['nu'])
            nu0 = np.asarray(bandsh_beams["nu_0"])
            alpha = np.asarray(bandsh_beams["alpha"])
            blT = self.beam_interpolation(b[:self.l_bpws[-1]+1], np.ones(self.l_bpws[-1]+1), np.arange(self.l_bpws[-1]+1), nu, nu0, alpha)

            bandsh_beams = self.bandpass_shifted_beams[f"{exp}_s2"]
            #reading the Delta nu list in the file
            dnulist = np.array([float(dn) for dn in bandsh_beams["beams"].keys()])
            #finding the Delta nu closer to the sampled one
            Dnu = dnulist[abs(dnulist - dnu).argmin()]
            #reading the corresponding monochromatic beam 
            b = bandsh_beams["beams"][f"{Dnu}"]
            #using the same freq array as the bandpass one
            # nu pol should be the same as the T one, I'll comment it for now
            # nu = np.asarray(self.bands[f"{exp}_s2"]['nu'])
            nu0 = np.asarray(bandsh_beams["nu_0"])
            alpha = np.asarray(bandsh_beams["alpha"])
            blP = self.beam_interpolation(b[:self.l_bpws[-1]+1], np.ones(self.l_bpws[-1]+1), np.arange(self.l_bpws[-1]+1), nu, nu0, alpha)
        else:
            #print("using beams with bandpass shift = 0")
            blT = self.beams[f"{exp}_s0"]["beams"]
            blP = self.beams[f"{exp}_s2"]["beams"]

        # normalizing the pol beam by the T one for each freq
        # if already normalized, this operation has no effect
        blP /= blT[:, 0][..., np.newaxis]
        # normalizing the beam profile such that it has a max at 1 at ell = 0
        blT /= blT[:, 0][..., np.newaxis]
        
        return blT[:,2:self.l_bpws[-1] + 1], blP[:,2:self.l_bpws[-1] + 1]
