import numpy as np

# analysis params
T0 = 15.0

# directories
rawdir = '/home/ckampa/Dropbox/LogFiles/'
pkldir = ('/home/ckampa/data/hallprobecalib_extras/datafiles/magnet_ramp/'+
          '2021-02-24/')
plotdir = ('/home/ckampa/data/hallprobecalib_extras/plots/magnet_ramp/'+
           '2021-02-24/')

# file names
# slow controls
rawfile = rawdir+'2021-02-24 094822slow.txt'
pklraw = pkldir+'ramp_2021-02-24.raw.pkl'
pklinfo = pkldir+'ramp_2021-02-24.run-info.pkl'
pklinfo_hall_regress = pkldir+'ramp_2021-02-24.run-info_hall_regression.pkl'
pklinfo_nmr_regress = pkldir+'ramp_2021-02-24.run-info_nmr_regression.pkl'
pklproc = pkldir+'ramp_2021-02-24.processed-all.pkl'
pklproc_ramp = pkldir+'ramp_2021-02-24.processed-ramp.pkl'
pklproc_hyst = pkldir+'ramp_2021-02-24.processed-hyst.pkl'
# fit pickles
# processing
pklfit_stable_temp = pkldir+'ramp_2021-02-24.stable-temp-fits.pkl'
pklfit_temp_nmr = pkldir+'ramp_2021-02-24.nmr-lin-temp-regress-fits.pkl'
pklfit_temp_hall = pkldir+'ramp_2021-02-24.hall-lin-temp-regress-fits.pkl'
pklfit_temp_hall_nmr = (pkldir+'ramp_2021-02-24.'+
                        'hall-from-nmr-lin-temp-regress-fits.pkl')
# fit results
pkl_interp_fcn_NMR = pkldir+'ramp_2021-02-24.interp_fcn_NMR.pkl'
pkl_interp_fcn_Hall = pkldir+'ramp_2021-02-24.interp_fcn_Hall.pkl'
# LaTeX output
tex_info = plotdir+'final_results/run_info.tex'
# FEMM
femmfile_70 = pkldir+'gap70_B_vs_I_r0z0_0-200_results.txt'
femmfile_70_1006 = pkldir+'gap70_z0_1006_Steel_results.txt'
femmfile_75_Hall = pkldir+'gap75_B_vs_I_r0z0_0-300_results.txt'
femmfile_75_NMR = pkldir+'gap75_B_vs_I_r0z37.5_results.txt'
femmfile_75_estimate = pkldir+'ramp_2021-02-24_guess_gap75_B_vs_I_results.txt'
femmfile_75_1006 = (pkldir+'gap75_1006_Steel_ramp_2021-02-24_ideal_B_vs_I'+
                    '_results.txt')
# GMW excitation data -- gap = 70 mm, x=y=z=0
GMW_currents = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
GMW_Bs = np.array([0.00000,0.25612,0.50921,0.76165,0.99680,1.14767,1.25025,
                  1.35284,1.42225,1.47993,1.51854])
# current set points (approx.) -- use for FEMM test fits
# Hall_currents = np.array([0., 32., 64., 96., 128., 144., 160., 176., 192.,
#                           208., 224., 240., 256., 272., 281.])
# FIXME! 281. not 280.
Hall_currents = np.array([0., 32., 64., 96., 128., 144., 160., 176., 192.,
                          208., 224., 240., 256., 272., 280.])
# NMR_currents = np.concatenate([[0.], Hall_currents[4:]])
NMR_currents = np.concatenate([Hall_currents[4:]])

# Other params
probe = '6A0000000D61333A' # on SmarAct, in magnet
