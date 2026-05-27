import os
import re
import numpy as np

def get_step_from_filename(path):
    """Pull the first integer out of the file's basename (no extension)."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else 0

def normalise_symbol(sym):
    return sym[0].upper() + sym[1:].lower()

def read_xyz(path):
    """Plain xyz reader, returns (symbols, coords)."""
    lines  = path.read_text().splitlines() if hasattr(path, 'read_text') else open(path).read().splitlines()
    natoms = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        sym = normalise_symbol(parts[0])
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        symbols.append(sym)
        coords.append(xyz)
    return symbols, np.array(coords, dtype=float)

