import numpy as np
import pandas as pd

# GLOBALS
ADC_per_mm = 2560.
mm_per_ADC = 1./ADC_per_mm

x0_ADC = 422190; x0 = xwidth = 12.90 - 2 # mm, digital calipers accounts for NMR sample depth
y0_ADC = 416169; y0 = 124.6 # mm, from triangulation
z0_ADC = 1649035; z0 = 124.5 # mm, from triangulation

ADC_dict = {'X': x0_ADC, 'Y': y0_ADC, 'Z': z0_ADC}
mag_mm_dict = {'X': x0, 'Y': y0, 'Z': z0}

def ADC_to_mm(ADC, coord):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    delta_ADC = sf*(ADC - ADC_dict[coord])
    return mag_mm_dict[coord] + delta_ADC * mm_per_ADC
    
def mm_to_ADC(mm, coord):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    delta_mm = sf*(mm - mag_mm_dict[coord])
    return ADC_dict[coord] + delta_mm * ADC_per_mm
