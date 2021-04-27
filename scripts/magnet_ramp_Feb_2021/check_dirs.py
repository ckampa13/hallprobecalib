import os
from datetime import datetime
from configs import pkldir, plotdir

time0 = datetime.now()

p_0 = plotdir+'time/'
p_0_0 = plotdir+'time/all/'
p_1 = plotdir+'final_results/'
p_1_0 = plotdir+'final_results/stable_temp/'
p_1_1 = plotdir+'final_results/nmr_temp_regress/'
p_1_2 = plotdir+'final_results/hall_temp_regress/'
p_1_3 = plotdir+'final_results/hall_from_nmr_temp_regress/'
p_2 = plotdir+'analysis_guides/'

dirs = [pkldir, plotdir,
        p_0, p_0_0, p_1, p_1_0, p_1_1, p_1_2, p_1_3,
        p_2,
       ]

print('Running script: check_dirs.py')
#print('Checking Directories:')

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f'Created: {d}')

timef = datetime.now()

#print('Done checking directories.\n')
print(f'Runtime: {timef-time0} [H:MM:SS])\n')
