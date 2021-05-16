import numpy as np
import pandas as pd

# 250 mm pole face, "data" from Magneto simulation software
# current settings
# GMW_currents stored in configs.py, move here? FIXME!
GMW_currents = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
# B at different gaps
GMW_60_Bs = np.array([0.0000, 0.29859, 0.59333, 0.88187, 1.12580, 1.26766,
                      1.36909, 1.46625, 1.52341, 1.57538, 1.61025])
GMW_70_Bs = np.array([0.00000, 0.25612, 0.50921, 0.76165, 0.99680, 1.14767,
                      1.25025, 1.35284, 1.42225, 1.47993, 1.51854])
GMW_80_Bs = np.array([0.00000, 0.22425, 0.44503, 0.66931, 0.88192, 1.04865,
                      1.15564, 1.24578, 1.32747, 1.39045, 1.43610])
GMW_100_Bs = np.array([0.00000, 0.17943, 0.35864, 0.53709, 0.71592, 0.86692,
                       0.99263, 1.07621, 1.15447, 1.22117, 1.27787])

df_GMW = pd.DataFrame({'I':2*GMW_currents, 'B_60':GMW_60_Bs, 'B_70':GMW_70_Bs,
                       'B_80':GMW_80_Bs, 'B_100':GMW_100_Bs})
