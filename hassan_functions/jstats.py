import numpy as np

# "Cubic" (power-mean) average used to collapse an ensemble of |J| couplings into a
# single effective value comparable to experiment. The exponent up-weights the larger
# couplings, which dominate the measured signal. p=2.25 is the reference used throughout.
CUBIC_P = 2.25

def cubic_mean(vals, p=CUBIC_P):
    return float((np.mean(vals**p))**(1./p))

def cubic_dispersion(vals, p=CUBIC_P):
    cm = cubic_mean(vals, p=p)
    return float((np.mean(np.abs(vals - cm)**p))**(1./p))
