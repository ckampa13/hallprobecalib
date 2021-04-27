#!/bin/bash
# Run magnet ramp analysis scripts

source /home/ckampa/anaconda3/etc/profile.d/conda.sh
conda activate mu2e

# check appropriate directories exist
python check_dirs.py
# FEMM / GMW comparison plot
#python femm_gmw_compare.py
# preprocess data
#python preprocess_data.py
# process data (temperature regression)
# NEEDS HANDLING HALL PROBES
python process_data_temp_regress.py
# B vs. I with temperature regressed data
# python B_vs_I_no_temp.py

# read -p "Press any key to resume ..."
