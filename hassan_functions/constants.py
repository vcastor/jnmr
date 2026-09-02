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

# ── named NMR sites ──────────────────────────────────────────────────────────
# choline is (CH3)3N(+)-CH2-CH2-OH, urea is H2N-CO-NH2.
SITE_LABELS = {
    'H1':    {'species': 'choline', 'symbol': 'H',
              'group': 'CH3', 'multiplicity': 9,
              'description': 'H on a methyl carbon of the trimethylammonium head'},
    'H2':    {'species': 'choline', 'symbol': 'H',
              'group': 'CH2 bonded to N+', 'multiplicity': 2,
              'description': 'H on the CH2 next to the quaternary N'},
    'H3':    {'species': 'choline', 'symbol': 'H',
              'group': 'CH2 bonded to O', 'multiplicity': 2,
              'description': 'H on the CH2 next to the hydroxyl O'},
    'H4':    {'species': 'choline', 'symbol': 'H',
              'group': 'OH', 'multiplicity': 1,
              'description': 'hydroxyl H'},
    'H5':    {'species': 'urea', 'symbol': 'H',
              'group': 'NH2', 'multiplicity': 4,
              'description': 'amine H — the only H in urea'},
    'Nurea': {'species': 'urea', 'symbol': 'N',
              'group': 'NH2', 'multiplicity': 2,
              'description': 'amide N of urea'},
}

SITE_COUPLINGS = {
    'H1-H2':    {'sites': ('H1', 'H2'),    'scopes': ('intra',)},
    'H1-H3':    {'sites': ('H1', 'H3'),    'scopes': ('intra', 'inter')},
    'H1-H4':    {'sites': ('H1', 'H4'),    'scopes': ('intra', 'inter')},
    'H5-H2':    {'sites': ('H5', 'H2'),    'scopes': ('inter',)},
    'H5-H3':    {'sites': ('H5', 'H3'),    'scopes': ('inter',)},
    'H5-H4':    {'sites': ('H5', 'H4'),    'scopes': ('inter',)},
    'Nurea-H1': {'sites': ('Nurea', 'H1'), 'scopes': ('inter',)},
    'Nurea-H2': {'sites': ('Nurea', 'H2'), 'scopes': ('inter',)},
    'Nurea-H3': {'sites': ('Nurea', 'H3'), 'scopes': ('inter',)},
}

def pair_type(site_a, site_b):
    """The SITE_COUPLINGS key for two site labels, or None if the pair is not one of
    the requested couplings. Order-insensitive: the reader classifies a (pert, resp)
    pair from geometry and does not know which way round the team writes it."""
    for key, spec in SITE_COUPLINGS.items():
        if set(spec['sites']) == {site_a, site_b}:
            return key
    return None

