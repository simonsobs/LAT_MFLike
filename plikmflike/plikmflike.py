from cobaya.likelihood import Likelihood
import numpy as np
import sys
from .fgspectra import cross as fgc
from .fgspectra import power as fgp
from .fgspectra import frequency as fgf

class PlikMFLike(Likelihood):
	def initialize(self):
		self.expected_params = [
			'cib_index', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ksz_norm',
			'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217',
			# These parameters aren't used, but they are kept for backwards compatibility.
			'A_sbpx_100_100_TT', 'A_sbpx_143_143_TT', 'A_sbpx_143_217_TT', 'A_sbpx_217_217_TT',
			
			'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217',
			
			'galf_TE_index',
			'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217',
			
			'galf_EE_index',
			'galf_EE_A_100', 'galf_EE_A_100_143', 'galf_EE_A_100_217', 'galf_EE_A_143', 'galf_EE_A_143_217', 'galf_EE_A_217',
			
			'A_cnoise_e2e_100_100_EE', 'A_cnoise_e2e_100_143_EE', 'A_cnoise_e2e_100_217_EE', 'A_cnoise_e2e_143_143_EE', 'A_cnoise_e2e_143_217_EE', 'A_cnoise_e2e_217_217_EE',
			
			'calib_100T', 'calib_217T', 'calib_100P', 'calib_143P', 'calib_217P', 'A_planck'
		]
		
		self.prepare_data()
	
	def prepare_data(self):
		self.nmin = [ [1,1,1,1], [1,1,60,1,60,60], [1,1,60,1,60,60] ]
		self.nmax = [ [136,199,215,215], [114,114,114,199,199,199], [114,114,114,199,199,199] ]
		
		self.nbintt = [ b - a + 1 for a, b in zip(self.nmin[0], self.nmax[0]) ]
		self.nbinte = [ b - a + 1 for a, b in zip(self.nmin[1], self.nmax[1]) ]
		self.nbinee = [ b - a + 1 for a, b in zip(self.nmin[2], self.nmax[2]) ]
		
		self.crosstt = [ (0,0), (1,1), (1,2), (2,2) ]
		self.crosste = [ (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ]
		self.crossee = [ (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ]
		
		self.freqs = [ 100, 143, 217 ]
		
		self.sys_vec = None
		self.inv_cov = None
		
		self.b_ell = None
		self.b_dat = None
		self.win_func = None
		
		self.log.debug('Loading windows.')
		self.load_windows_pliklike(self.weightfile, self.minfile, self.maxfile, self.nmin, self.nmax, data_dir = self.data_folder)
		
		self.log.debug('Loading inv cov.')
		self.inv_cov = np.loadtxt(self.data_folder + self.covfile, dtype = float)[:self.nbin,:self.nbin]
		
		self.log.debug('Loading spectrum data.')
		self.b_dat = np.loadtxt(self.data_folder + self.specfile, dtype = float)[:self.nbin, 1]
		
		self.log.debug('Loading systematics.')
		self.load_systematics(self.leakfile, self.corrfile, self.subpixfile, data_dir = self.data_folder)
		
		self.log.debug('Done preparing all data!')
	
	def load_windows_pliklike(self, weightfile, minfile, maxfile, bin_starts, bin_ends, data_dir = ''):
		# Because of the way the plik files store the window function, I wrote this function to load in the window function into a matrix form.
		# It's not the nicest code I have ever written, but it does what it needs to do.
		# For optimal use, call this function once, output the resulting win_func to a text file, and then load that in using load_plaintext every time.
		blmin = np.loadtxt(data_dir + minfile).astype(int) + self.lmin
		blmax = np.loadtxt(data_dir + maxfile).astype(int) + self.lmin
		bweight = np.concatenate([ np.zeros((self.lmin-1)), np.loadtxt(data_dir + weightfile) ])
		
		blens = [ [ b - a + 1 for a, b in zip(x, y) ] for x, y in zip(bin_starts, bin_ends) ]
		bweight = np.repeat(bweight[np.newaxis,:], max(blens[0]), axis = 0)
		
		# Basically, bweight temporarily stores the full window function, and we will take slices from it and put that in the full window function.
		for i in np.arange(bweight.shape[0]):
			bweight[i, :blmin[i]-1] = 0.0
			bweight[i, blmax[i]:] = 0.0
		
		xmin = []
		xmax = []
		for a, b in zip(bin_starts, bin_ends):
			xmin += a
			xmax += b
		
		xmin = np.array(xmin) - 1
		xmax = np.array(xmax)
		xlen = xmax - xmin
		
		self.win_func = np.zeros((sum([ sum(x) for x in blens ]), self.shape))
		
		for i in np.arange(len(xmin)):
			xstart = np.sum(xlen[0:i])
			xend = xstart + xlen[i]
			
			self.win_func[xstart:xend, :] = bweight[xmin[i]:xmax[i],1:self.shape + 1]
		
		del bweight
		
		self.nmin = bin_starts
		self.nmax = bin_ends
		self.b_ell = self.win_func @ np.arange(2, self.lmax_win + 1)
	
	def load_systematics(self, leak_filename, corr_filename, subpix_filename, data_dir = ''):
		leakage = np.loadtxt(data_dir + leak_filename)[:,1:]
		corr = np.loadtxt(data_dir + corr_filename)[:,1:]
		subpix = np.loadtxt(data_dir + subpix_filename)[:,1:]
		
		sum_vec = (leakage + corr + subpix)
		sys_vec = np.zeros((self.shape, sum_vec.shape[1]))
		
		sys_vec[:sum_vec.shape[0],:] = sum_vec[:,:]
		
		sys_vec = self.win_func @ sys_vec
		
		self.sys_vec = np.zeros((self.win_func.shape[0]))
		
		k = 0
		for j, tt in enumerate(self.nbintt):
			self.sys_vec[k:k+tt] = sys_vec[k:k+tt,j]
			k += tt
		
		# The sys vector is sorted TT-TE-EE, but it should be sorted TT-EE-TE, so we swap ordering here a bit.
		k = 0
		k0 = sum(self.nbintt)
		k1 = sum(self.nbintt) + sum(self.nbinte)
		j1 = len(self.nbintt) + len(self.nbinte)
		for j, ee in enumerate(self.nbinee):
			self.sys_vec[k+k0:k+k0+ee] = sys_vec[k+k1:k+k1+ee,j+j1]
			k += ee
		
		k = 0
		k0 = sum(self.nbintt) + sum(self.nbinee)
		k1 = sum(self.nbintt)
		j1 = len(self.nbintt)
		for j, te in enumerate(self.nbinte):
			self.sys_vec[k+k0:k+k0+te] = sys_vec[k+k1:k+k1+te,j+j1]
			k += te
	
	def get_requirements(self):
		return {
			'Cl' : {
				'tt' : self.tt_lmax,
				'te' : self.tt_lmax,
				'ee' : self.tt_lmax
			}
		}
	
	def get_model(self, cl, **params_values):
		self.log.debug('Start calculating model.')
		l0 = int(2 - cl['ell'][0])
		ls = cl['ell'][l0:self.shape+l0]
		cl_tt = cl['tt'][l0:self.shape+l0]
		cl_te = cl['te'][l0:self.shape+l0]
		cl_ee = cl['ee'][l0:self.shape+l0]
		
		fg = get_Planck_foreground(params_values, ls)
		
		fg_tt = np.zeros((self.nspectt, self.shape))
		fg_te = np.zeros((self.nspecte, self.shape))
		fg_ee = np.zeros((self.nspecee, self.shape))

		for i, (c1, c2) in enumerate(self.crosstt):
			f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
			fg_tt[i, :] = fg['tt', 'all', f1, f2][:self.shape]

		for i, (c1, c2) in enumerate(self.crosste):
			f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
			fg_te[i, :] = fg['te', 'all', f1, f2][:self.shape]

		for i, (c1, c2) in enumerate(self.crossee):
			f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
			fg_ee[i, :] = fg['ee', 'all', f1, f2][:self.shape]
		
		self.log.debug('Summing theory = CMB + foreground.')
		
		x_theory = np.zeros((self.nspec, self.shape))
		
		x_theory[0                         : self.nspectt                          ,:self.shape] = np.tile(cl_tt, (self.nspectt, 1)) + fg_tt
		x_theory[self.nspectt              : self.nspectt+self.nspecee             ,:self.shape] = np.tile(cl_ee, (self.nspecee, 1)) + fg_ee
		x_theory[self.nspectt+self.nspecee : self.nspectt+self.nspecee+self.nspecte,:self.shape] = np.tile(cl_te, (self.nspecte, 1)) + fg_te
		
		self.log.debug('Completed theory vector. Now binning.')
		
		x_model = np.zeros((self.nbin))
		
		# TT modes
		for j in range(self.nspectt):
			x_model[sum(self.nbintt[0:j]) : sum(self.nbintt[0:j+1])] = self.win_func[sum(self.nbintt[0:j]) : sum(self.nbintt[0:j+1]), :] @ x_theory[j,:] # TT
		
		# EE modes
		for j in range(self.nspecee):
			i0 = sum(self.nbintt)
			j0 = self.nspectt
			x_model[i0 + sum(self.nbinee[0:j]) : i0 + sum(self.nbinee[0:j+1])] = self.win_func[i0 + sum(self.nbinee[0:j]) : i0 + sum(self.nbinee[0:j+1]), :] @ x_theory[j0+j,:] # EE
		
		# TE modes
		for j in range(self.nspecte):
			i0 = sum(self.nbintt) + sum(self.nbinee)
			j0 = self.nspectt + self.nspecee
			x_model[i0 + sum(self.nbinte[0:j]) : i0 + sum(self.nbinte[0:j+1])] = self.win_func[i0 + sum(self.nbinte[0:j]) : i0 + sum(self.nbinte[0:j+1]), :] @ x_theory[j0+j,:] # TE
		
		# x = x / [ l (l + 1) / 2 pi ]
		ll = np.arange(self.shape) + 2
		ell_factor = (ll.astype(float) * (ll + 1.0)) / (2.0 * np.pi)
		x_model = x_model / (self.win_func @ ell_factor)
		
		self.log.debug('Adding systematics.')
		
		x_model += self.sys_vec
		
		self.log.debug('Calibrating.')
		
		ct = 1.0 / np.sqrt(np.array([ params_values['calib_100T'], 1.0, params_values['calib_217T'] ]))
		yp = 1.0 / np.sqrt(np.array([ params_values['calib_100P'], params_values['calib_143P'], params_values['calib_217P'] ]))
		
		# Calibration
		for i in np.arange(len(self.nbintt)):
			# Mode T[i]xT[j] should be calibrated using CT[i] * CT[j]
			m1, m2 = self.crosstt[i]
			x_model[ sum(self.nbintt[0:i]) : sum(self.nbintt[0:i+1]) ] = x_model[ sum(self.nbintt[0:i]) : sum(self.nbintt[0:i+1]) ] * ct[m1] * ct[m2]
		
		for i in np.arange(len(self.nbinee)):
			# Mode E[i]xE[j] should be calibrated using (CT[i]*YP[i]) * (CT[j]*YP[j])
			m1, m2 = self.crossee[i]
			i0 = sum(self.nbintt)
			x_model[ i0 + sum(self.nbinee[0:i]) : i0 + sum(self.nbinee[0:i+1]) ] = x_model[ i0 + sum(self.nbinee[0:i]) : i0 + sum(self.nbinee[0:i+1]) ] * (ct[m1] * yp[m1]) * (ct[m2] * yp[m2])
		
		for i in np.arange(len(self.nbinte)):
			# Mode T[i]xE[j] should be calibrated using CT[i] * (CT[j]*YP[j])
			m1, m2 = self.crosste[i]
			i0 = sum(self.nbintt) + sum(self.nbinee)
			x_model[ i0 + sum(self.nbinte[0:i]) : i0 + sum(self.nbinte[0:i+1]) ] = x_model[ i0 + sum(self.nbinte[0:i]) : i0 + sum(self.nbinte[0:i+1]) ] * (0.5 * ct[m1] * (ct[m2] * yp[m2]) + 0.5 * (ct[m1] * yp[m1]) * ct[m2])
		
		# Calibrating for the overall Planck calibration parameter.
		x_model /= (params_values['A_planck'] ** 2.0)
		
		self.log.debug('Done calculating model.')
		
		return x_model
	
	def logp(self, **params_values):
		cl = self.theory.get_Cl(ell_factor = True)
		return self.loglike(cl, **params_values)
	
	def loglike(self, cl, **params_values):
		x_model = self.get_model(cl, **params_values)
		diff_vec = self.b_dat - x_model
		
		tmp = self.inv_cov @ diff_vec
		return -0.5 * np.dot(tmp, diff_vec)
	
	@property
	def use_tt(self):
		return self.enable_tt
	
	@use_tt.setter
	def use_tt(self, val):
		self.enable_tt = val
	
	@property
	def use_te(self):
		return self.enable_te
	
	@use_te.setter
	def use_te(self, val):
		self.enable_te = val
	
	@property
	def use_ee(self):
		return self.enable_ee
	
	@use_ee.setter
	def use_ee(self, val):
		self.enable_ee = val
	
	@property
	def frequencies(self):
		return self.freqs
	
	@property
	def nspectt(self):
		return len(self.nbintt)
	
	@property
	def nspecte(self):
		return len(self.nbinte)
	
	@property
	def nspecee(self):
		return len(self.nbinee)
	
	@property
	def nbin(self):
		# total number of bins
		return sum(self.nbintt) + sum(self.nbinte) + sum(self.nbinee)
	
	@property
	def nspec(self):
		# total number of spectra
		return self.nspectt + self.nspecte + self.nspecee
	
	@property
	def shape(self):
		return self.lmax_win-1
	
	@property
	def input_shape(self):
		return self.tt_lmax-1

# The spectra templates for the foregrounds.
ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_Planck())
tsz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.tSZ_Planck())
cib = fgc.PlankCrossSpectrum(fgf.ConstantSED(), fgp.CIB_Planck())
ttps = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.SquarePowerLaw())
tszxcib = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.tSZxCIB_Planck())
gal = fgc.PlankCrossSpectrum(fgf.ConstantSED(), fgp.gal_Planck())
galte = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerLaw())

def get_Planck_foreground(fg_params, ell, requested_cls = ['tt', 'te', 'ee']):
	frequencies = np.asarray([100, 143, 217], dtype=int)
	
	nu_0 = 150.0
	ell_0 = 3000
	
	tSZcorr = np.array([ 2.022, 0.95, 0.0000476 ])
	CIBcorr = np.array([ 0.0, 0.094, 1.0 ])
	
	model = { }
	
	tsz_amp = np.zeros((len(frequencies), len(frequencies)))
	tsz_amp[0,0] = fg_params['A_sz'] * tSZcorr[0]
	tsz_amp[1,1] = fg_params['A_sz'] * tSZcorr[1]
	tsz_amp[1,2] = fg_params['A_sz'] * np.sqrt(tSZcorr[2])
	tsz_amp[2,2] = fg_params['A_sz'] * tSZcorr[2]
	
	ps_amp = np.zeros((len(frequencies), len(frequencies)))
	ps_amp[0,0] = fg_params['ps_A_100_100']
	ps_amp[1,1] = fg_params['ps_A_143_143']
	ps_amp[1,2] = fg_params['ps_A_143_217']
	ps_amp[2,2] = fg_params['ps_A_217_217']
	
	gal_amp = np.zeros((len(frequencies), len(frequencies)))
	gal_amp[0,0] = fg_params['gal545_A_100']
	gal_amp[1,1] = fg_params['gal545_A_143']
	gal_amp[1,2] = fg_params['gal545_A_143_217']
	gal_amp[2,1] = fg_params['gal545_A_143_217']
	gal_amp[2,2] = fg_params['gal545_A_217']
	
	galte_amp = np.zeros((len(frequencies), len(frequencies)))
	galte_amp[0,0] = fg_params['galf_TE_A_100']
	galte_amp[0,1] = fg_params['galf_TE_A_100_143']
	galte_amp[0,2] = fg_params['galf_TE_A_100_217']
	galte_amp[1,0] = fg_params['galf_TE_A_100_143']
	galte_amp[2,0] = fg_params['galf_TE_A_100_217']
	galte_amp[1,1] = fg_params['galf_TE_A_143']
	galte_amp[1,2] = fg_params['galf_TE_A_143_217']
	galte_amp[2,1] = fg_params['galf_TE_A_143_217']
	galte_amp[2,2] = fg_params['galf_TE_A_217']
	
	galee_amp = np.zeros((len(frequencies), len(frequencies)))
	galee_amp[0,0] = fg_params['galf_EE_A_100']
	galee_amp[0,1] = fg_params['galf_EE_A_100_143']
	galee_amp[0,2] = fg_params['galf_EE_A_100_217']
	galee_amp[1,0] = fg_params['galf_EE_A_100_143']
	galee_amp[2,0] = fg_params['galf_EE_A_100_217']
	galee_amp[1,1] = fg_params['galf_EE_A_143']
	galee_amp[1,2] = fg_params['galf_EE_A_143_217']
	galee_amp[2,1] = fg_params['galf_EE_A_143_217']
	galee_amp[2,2] = fg_params['galf_EE_A_217']
	
	szcib_amp = np.zeros((len(frequencies), len(frequencies)))
	szcib_amp[0,0] = -2.0 * fg_params['xi_sz_cib'] * np.sqrt(fg_params['A_sz'] * tSZcorr[0] * fg_params['A_cib_217'] * CIBcorr[0])
	szcib_amp[1,1] = -2.0 * fg_params['xi_sz_cib'] * np.sqrt(fg_params['A_sz'] * tSZcorr[1] * fg_params['A_cib_217'] * CIBcorr[1])
	szcib_amp[1,2] = -fg_params['xi_sz_cib'] * np.sqrt(fg_params['A_sz'] * tSZcorr[1] * fg_params['A_cib_217'] * CIBcorr[2]) - fg_params['xi_sz_cib'] * np.sqrt(fg_params['A_sz'] * tSZcorr[2] * fg_params['A_cib_217'] * CIBcorr[1])
	szcib_amp[2,2] = -2.0 * fg_params['xi_sz_cib'] * np.sqrt(fg_params['A_sz'] * tSZcorr[2] * fg_params['A_cib_217'] * CIBcorr[2])
	
	# We keep it specific so yes this is all summed manually!
	model['tt', 'kSZ'] = fg_params['ksz_norm'] * ksz({'nu' : frequencies}, {'ell' : ell, 'ell_0' : ell_0})
	model['tt', 'tSZ'] = tsz_amp[...,np.newaxis] * tsz({"nu": frequencies}, {"ell": ell, "ell_0": ell_0})
	model['tt', 'tSZxCIB'] = szcib_amp[...,np.newaxis] * tszxcib({"nu": frequencies}, {"ell": ell, "ell_0": ell_0})
	model['tt', 'ps'] = ps_amp[...,np.newaxis] * ttps({"nu" : frequencies, "nu_0" : frequencies, "beta" : 0.0}, {"ell" : ell, "ell_0" : ell_0})
	model['tt', 'CIB'] = fg_params['A_cib_217'] * cib({"nu": frequencies}, {"ell": ell, "ell_0" : ell_0, 'n_cib' : fg_params['cib_index']})
	model['tt', 'gal'] = gal_amp[...,np.newaxis,np.newaxis] * gal({"nu" : frequencies}, {"ell" : ell})
	
	model['te', 'gal'] = galte_amp[...,np.newaxis] * galte({"nu" : frequencies}, {"ell" : ell, "ell_0" : 500.0, "alpha" : fg_params["galf_TE_index"] + 2.0})
	model['ee', 'gal'] = galee_amp[...,np.newaxis] * galte({"nu" : frequencies}, {"ell" : ell, "ell_0" : 500.0, "alpha" : fg_params["galf_EE_index"] + 2.0})
	
	fg_dict = { }
	
	for idx, (i, j) in enumerate([(0,0), (1,1), (1,2), (2,2)]):
		f1, f2 = frequencies[i], frequencies[j]
		
		fg_dict['tt', 'kSZ', f1, f2] = model['tt', 'kSZ'][i,j]
		fg_dict['tt', 'tSZ', f1, f2] = model['tt', 'tSZ'][i,j]
		fg_dict['tt', 'tSZxCIB', f1, f2] = model['tt', 'tSZxCIB'][i,j]
		fg_dict['tt', 'ps', f1, f2] = model['tt', 'ps'][i,j]
		fg_dict['tt', 'CIB', f1, f2] = model['tt', 'CIB'][i,j][:,idx] # Picking the right template.
		fg_dict['tt', 'gal', f1, f2] = model['tt', 'gal'][i,j][:,idx]
	
	for i, f1 in enumerate(frequencies):
		for j, f2 in enumerate(frequencies):
			fg_dict['te', 'gal', f1, f2] = model['te', 'gal'][i,j]
			fg_dict['ee', 'gal', f1, f2] = model['ee', 'gal'][i,j]
	
	component_list = {'tt' : ['kSZ', 'tSZ', 'tSZxCIB', 'CIB', 'gal', 'ps'], 'te' : ['gal'], 'ee' : ['gal']}
	for c1, f1 in enumerate(frequencies):
		for c2, f2 in enumerate(frequencies):
			for s in requested_cls:
				fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
				for comp in component_list[s]:
					if (s, comp, f1, f2) in fg_dict:
						fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]
	
	return fg_dict
