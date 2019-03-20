import numpy as np
import sacc

datadir = 'actpol_2f_full_s1315_sim_v3_forpy/data/data_act/2019/shared/'

tracer_names = np.array(["A90", "A150"])
exp_names = np.array(["ACT", "ACT"])

#corr_ordering = np.array([['A90_T', 'A90_T'], ['A90_T', 'A150_T'], ['A150_T', 'A150_T'], ['A90_T', 'A90_E'], ['A90_T', 'A150_E'], ['A90_E', 'A150_T'], ['A150_T', 'A150_E'], ['A90_E', 'A90_E'], ['A90_E', 'A150_E'], ['A150_E', 'A150_E']])
corr_ordering = np.array([['A90_T', 'A90_T'], ['A90_T', 'A150_T'], ['A150_T', 'A150_T'], ['A90_T', 'A90_E'], ['A90_T', 'A150_E'], ['A150_T', 'A90_E'], ['A150_T', 'A150_E'], ['A90_E', 'A90_E'], ['A90_E', 'A150_E'], ['A150_E', 'A150_E']])

def get_tracer_from_name(name, exp_sample=None):
    if name=='A90':
        nu = [90.]
    if name=='A150':
        nu = [150.]
    bandpass = [1.] 
    return sacc.Tracer(name, "spin2", np.asarray(nu), np.asarray(bandpass), exp_sample)

#Tracers
tracers=[get_tracer_from_name(t,e) for t,e in zip(tracer_names,exp_names)]

#Mean vector
dv = np.loadtxt(datadir + 'mr3c_20181012_190130_TT_TE_EE_C_ell_iter0.dat')[:, 1]
meanvec = sacc.MeanVec(dv)

#Precision matrix
cov = np.loadtxt(datadir + 'mr3c_20181012_190130_TT_TE_EE_cov_diag_C_ell.dat')
precis = sacc.Precision(cov)

#Binning
windows = {}
windows['TT'] = np.loadtxt(datadir + 'TT_C_ell_bpwf_v2_lmin2.dat')
windows['TE'] = np.loadtxt(datadir + 'TE_C_ell_bpwf_v2_lmin2.dat')
windows['EE'] = np.loadtxt(datadir + 'EE_C_ell_bpwf_v2_lmin2.dat')

nells = windows['TT'].shape[0]
ellmax = windows['TT'].shape[1]
ls = np.arange(ellmax) + 2

typ_arr=[]
ls_arr=[]
t1_arr=[]
t2_arr=[]
q1_arr=[]
q2_arr=[]
w_arr=[]
for ic, c in enumerate(corr_ordering):
    s1, s2 = c
    tn1 = s1[:-2]
    q1 = s1[-1]
    t1 = np.where(tracer_names==tn1)[0][0]
    tn2 = s2[:-2]
    q2 = s2[-1]
    t2 = np.where(tracer_names==tn2)[0][0]
    typ = q1 + q2
    for b in range(nells) :
        w = windows[typ][b]
        lmean = np.sum(ls * w) / np.sum(w)
        win = sacc.Window(ls, w)
        ls_arr.append(lmean)
        w_arr.append(win)
    q1_arr += nells * [q1]
    q2_arr += nells * [q2]
    t1_arr += nells * [t1]
    t2_arr += nells * [t2]
    typ_arr += nells * [typ]
bins = sacc.Binning(typ_arr, ls_arr, t1_arr, q1_arr, t2_arr, q2_arr, windows=w_arr)

#SACC files
s = sacc.SACC(tracers, bins, mean=meanvec, precision=precis, meta={'data_name':'ACTPol_TT_TE_EE_analysis'})

#Save SACC file
s.saveToHDF("ACTPol_0.sacc")
s.printInfo()

