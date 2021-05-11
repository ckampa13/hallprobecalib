#!/bin/bash
# Run magnet ramp analysis scripts

source /home/ckampa/anaconda3/etc/profile.d/conda.sh
conda activate mu2e

# check appropriate directories exist
python check_dirs.py

# FEMM / GMW comparison plot
# python femm_gmw_compare.py

# FEMM fit vs. interpolation
# python femm_fits.py

# preprocess data
# python preprocess_data.py

# process data (temperature regression)
# python process_data_temp_regress.py

# B vs. I with temperature regressed data
python B_vs_I_no_temp.py

# copy results to Dropbox folder with write-up
# cp -r /home/ckampa/data/hallprobecalib_extras/plots/magnet_ramp/2021-02-24 /home/ckampa/Dropbox/research/deliverables/mu2e_analysis_notes/hall_probes/magnet_ramp_2021-02-24/figures/

# read -p "Press any key to resume ..."
