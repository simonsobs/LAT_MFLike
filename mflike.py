import numpy as np
from sacc.sacc import SACC

class MFLike:

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra, bandpasses, and window functions. 
        """
        data_fname = 'ACTPol_0.sacc'
        self.s = SACC.loadFromHDF(data_fname)
        self.order = self.s.sortTracers()

        #Keep only BB measurements
        #self.s.cullType(b'BB') # TODO: Modify if we want to use E

        #Collect bandpasses
        self.bpasses = []
        self.meannu = []
        for t in self.s.tracers:
            nu = t.z
            if len(nu) == 1:
                # delta function bandpasses 
                dnu = 1.
            else:
                dnu = np.zeros_like(nu)
                dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
                dnu[0] = nu[1] - nu[0]
                dnu[-1] = nu[-1] - nu[-2]
            bnu = t.Nz
            self.bpasses.append([nu, dnu, bnu])
            #self.meannu.append(np.sum(dnu*nu*bnu) / np.sum(dnu*bnu))

        #Get ell sampling and windows
        self.bpw_l = self.s.binning.windows[0].ls
        _, _, _, self.ell_b, _ = self.order[0]
        self.windows = s.binning.windows

        #Get power spectra and covariances
        #Store data
        self.data = self.s.mean.vector   
        self.covar = self.s.precision.getCovarianceMatrix()
        self.invcov = s.precision.getPrecisionMatrix()
        return


    def integrate_seds(self, params):
        fg_scaling = {}
        for key in self.fg_model.components:
            fg_scaling[key] = []

        for tn in range(self.nfreqs):
            nus = self.bpasses[tn][0]
            bpass = self.bpasses[tn][1]
            dnu = self.bpasses[tn][2]
            bpass_integration = bpass * dnu

            for key, component in self.fg_model.components.items(): 
                conv_rj = (nus / component['nu0'])**2

                sed_params = [] 
                for param in component['sed'].params:
                    pindx = self.parameters.param_index[param]
                    sed_params.append(params[pindx])
                
                fg_units = component['cmb_n0_norm'] / self.cmb_norm[tn]
                fg_sed_eval = component['sed'].eval(nus, *sed_params) * conv_rj
                fg_sed_int = np.dot(fg_sed_eval, bpass_integration) * fg_units
                fg_scaling[key].append(fg_sed_int)
        return fg_scaling

    def evaluate_power_spectra(self, params):
        fg_pspectra = {}
        for key, component in self.fg_model.components.items():
            pspec_params = []
            # TODO: generalize for different power spectrum models
            # should look like:
            # for param in power_spectrum_model: get param index (same as the SEDs)
            for param in component['spectrum_params']:
                pindx = self.parameters.param_index[param]
                pspec_params.append(params[pindx])
            fg_pspectra[key] = normed_plaw(self.bpw_l, *pspec_params)
        return fg_pspectra
    
    # setup foreground model here takes (f1, f2, ell, params)
    # param dict

    def model(self, params):
        # param list
        """
        Defines the total model and integrates over the bandpasses and windows. 
        """
        cmb_bmodes = params[0] * self.cmb_bbr + self.cmb_bblensing
        fg_scaling = self.integrate_seds(params)
        fg_p_spectra = self.evaluate_power_spectra(params)
        
        model_cls = []
        
        # zip param names, param list 

        for t1, t2, typ, ells, ndx in self.order:
            window = self.windows[ndx]
            model0 = fg_model(f1, f2, ells, typ, params_dict)
            
            model = np.dot(window, model)
            model_cls.append(model)

        return model_cls

    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood. 
        """
        model_cls = self.model(params)
        dx = self.data - model_cls
        return -0.5 * np.einsum('i, ij, j', dx, self.invcov, dx)



