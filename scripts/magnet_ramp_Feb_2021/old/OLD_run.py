### USE RUN.BASH
# check that proper directories exist -- create them if not
exec(open("check_dirs.py").read())
# figure 1
#exec(open("femm_gmw_compare.py").read())
# pre-processing data
exec(open("preprocess_data.py").read())
