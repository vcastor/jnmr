import numpy as np

FORMULAS = {
    'urea':     'CH4N2O',
    'choline':  'C5H14NO',
    'chloride': 'Cl',
    'water':    'H2O',
}

ISOTOPE_FOR_SYMBOL = {
    'H':  '1H',
    'C':  '13C',
    'N':  '15N',
    'O':  '17O',
    'Cl': '35Cl',
}

GAMMA = {
    '1H':   2.6752218744e8,
    '13C':  6.728284e7,
    '15N': -2.71261804e7,
    '17O': -3.62808e7,
    '35Cl': 2.624198e7,
}

MU0_OVER_4PI  = 1.0e-7
HBAR          = 1.054571817e-34
TWO_PI        = 2.0*np.pi
ANGSTROM_TO_M = 1.0e-10

