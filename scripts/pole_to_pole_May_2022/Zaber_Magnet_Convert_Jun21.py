# Globals
# found from triangulating magnet coordinates at a reference point
# & comparing to Zaber readout

# microns per ADC (aka slope)
mic_per_ADC = 1./2.56
ADC_per_mm = 2560.
mm_per_ADC = 1./ADC_per_mm
# triangulated "magnet coordinates" @ reference point
x0 = 12.90 - 2 # mm, digital calipers accounts for NMR sample depth
y0 = 124.6 # mm, from triangulation
z0 = 124.5 # mm, from triangulation
# ADC values @ reference point
x0_ADC = 422190
y0_ADC = 416169
z0_ADC = 1649035
# micron values @ reference point
x0_mic = x0_ADC * mic_per_ADC
y0_mic = y0_ADC * mic_per_ADC
z0_mic = z0_ADC * mic_per_ADC
# Zaber micron --> mm @ reference point
x0_mm = x0_mic * 1e-3
y0_mm = y0_mic * 1e-3
z0_mm = z0_mic * 1e-3
# low-field NMR offsets
dX_low_NMR = -3.88 # mm
dY_low_NMR = 0. # mm
dZ_low_NMR = 21.01 # mm

# some dictionaries for convenience
# starting from ADC
ADC_dict = {'X': x0_ADC, 'Y': y0_ADC, 'Z': z0_ADC}
mag_mm_dict = {'X': x0, 'Y': y0, 'Z': z0}
# starting from Micron
Micron_dict  = {'X': x0_mic, 'Y': y0_mic, 'Z': z0_mic}
# low-field NMR
offsets_lowfield_mm = {'X': dX_low_NMR, 'Y': dY_low_NMR, 'Z': dZ_low_NMR}

# functions
# Zaber ADC
def zaber_ADC_to_magnet_mm(ADC, coord, low_NMR=False):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    if low_NMR:
        d_low = offsets_lowfield_mm[coord]
    else:
        d_low = 0.
    delta_ADC = sf*(ADC - ADC_dict[coord])
    return mag_mm_dict[coord] + delta_ADC * mm_per_ADC + d_low

def magnet_mm_to_zaber_ADC(mm, coord, low_NMR=False):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    if low_NMR:
        d_low = offsets_lowfield_mm[coord]
    else:
        d_low = 0.
    delta_mm = sf*(mm - mag_mm_dict[coord] - d_low)
    return ADC_dict[coord] + delta_mm * ADC_per_mm

# Zaber Micron
def zaber_micron_to_magnet_mm(micron, coord, low_NMR=False):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    if low_NMR:
        d_low = offsets_lowfield_mm[coord]
    else:
        d_low = 0.
    d_mm = sf * (micron - Micron_dict[coord]) * 1e-3
    magnet_mm = mag_mm_dict[coord] + d_mm + d_low
    return magnet_mm

def magnet_mm_to_zaber_micron(mm, coord, low_NMR=False):
    if coord == 'Z':
        sf = -1
    else:
        sf = 1
    if low_NMR:
        d_low = offsets_lowfield_mm[coord]
    else:
        d_low = 0.
    d_mic = sf * (mm - mag_mm_dict[coord] - d_low) * 1e3
    zaber_mic = Micron_dict[coord] + d_mic
    return zaber_mic