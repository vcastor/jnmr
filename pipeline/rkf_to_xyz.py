#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np

def read_molecule_from_rkf(rkf_path: str, frame: int = -1) -> 'Molecule':
    """
    Read molecule from RKF file (MD trajectory checkpoint).
    frame: -1 for last frame, or specific frame number (1-based)
    """
    kf = KFFile(rkf_path)
    
    natoms  = kf.read('Molecule', 'nAtoms')
    symbols = kf.read('Molecule', 'AtomSymbols').split()
    
    if 'MDHistory' in kf.sections():
        nsteps = kf.read('MDHistory', 'nEntries')
        if frame == -1:
            frame = nsteps
        coords = np.array(kf.read('MDHistory', f'Coords({frame})')).reshape(natoms, 3)
        coords *= 0.529177
    else:
        coords = np.array(kf.read('Molecule', 'Coords')).reshape(natoms, 3)
        coords *= 0.529177
    
    mol = Molecule()
    for sym, xyz in zip(symbols, coords):
        mol.add_atom(Atom(symbol=sym, coords=tuple(xyz)))
    
    return mol

def rkf_to_xyz(rkf_path: str, xyz_path: str, frame: int = -1) -> None:
    mol = read_molecule_from_rkf(rkf_path, frame)
    mol.write(xyz_path)

# ── main ──────────────────────────────────────────────────────────────────
init()

config = {
    "input_dir": "mdSteps/rkf",
    "output_dir": "mdSteps/xyz",
}

rkf_files = sorted(glob.glob(os.path.join(config["input_dir"], "*.rkf")))

for rkf in rkf_files:
    basename = os.path.splitext(os.path.basename(rkf))[0]
    xyz = os.path.join(config["output_dir"], f"{basename}.xyz")
    
    if os.path.exists(xyz):
        continue
    
    rkf_to_xyz(rkf, xyz)

finish()

