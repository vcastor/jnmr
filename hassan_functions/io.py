import os
import re
import numpy as np

CHARGE_HEADER = "Atom         Population     Net charge    Spin density      Laplacian"

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

def read_labeled_matrix(path, header):
    """Symmetric (n, n) matrix from an AMS output block keyed by absolute atom
    numbers. The block must start at `header` and use rows of the form
    `<atom_n> <symbol>: v1 v2 ...` with the lower triangle on each row.

    Returns (atom_labels, matrix).
    """
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if header in l) + 3

    atom_labels = []; rows = []; i = start
    while i < len(lines):
        m = re.match(r"\s*(\d+)\s+\w+\s*:(.*)$", lines[i])
        if m is None:
            break
        atom_labels.append(int(m.group(1)))
        nvals = len(atom_labels)
        vals  = [float(x) for x in m.group(2).split()]
        i += 1
        while len(vals) < nvals:
            vals.extend(float(x) for x in lines[i].split())
            i += 1
        rows.append(vals[:nvals])

    n = len(atom_labels)
    matrix = np.zeros((n, n))
    for k, vals in enumerate(rows):
        matrix[k, :k + 1] = vals
        matrix[:k + 1, k] = vals
    return atom_labels, matrix

def read_qtaim_charges(path):
    """{atom_number: net_charge} from a QTAIM .out file.

    Block starts 2 lines after CHARGE_HEADER. Each data row has 7 whitespace
    fields: atom number at position 0, net charge at position 4. The table
    ends on the first line whose split() length differs from 7.
    """
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if CHARGE_HEADER in l) + 2
    charges = {}
    for line in lines[start:]:
        parts = line.split()
        if len(parts) != 7:
            break
        charges[int(parts[0])] = float(parts[4])
    return charges

