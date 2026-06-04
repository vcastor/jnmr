#!$AMSBIN/plams
import os
import glob
import numpy as np
from pathlib import Path
from hassan_functions.geometry import distance
from hassan_functions.finders import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.ordering import classify_sort, compute_offsets
from hassan_functions.constants import FORMULAS

CLUSTERS_DIR = "clusters"
SRC_DIR      = Path("run_scripts/TZ2P_FC")   # geometry source
OUT_DIR      = Path("run_scripts/c_h")
SPECIES      = ['urea', 'choline', 'chloride']

# Set to a float (Å) to restrict response H atoms to within that distance of C(urea).
# None = include all pairs regardless of distance.
DIST_THRESHOLD = None

# Restrict to specific MD step numbers; None = process every available snapshot.
SNAPSHOTS = {1180000}

PARTITION     = "tlong"
WALLTIME      = "199:00:00"
NTASKS        = 12
CPUS_PER_TASK = 16
MODULE        = "atomic_simu/cobra-ams/2025.106_amd"


def extract_atoms_block(run_path):
    """Return raw atom lines from the System/Atoms block of an AMS run file."""
    lines = []
    in_atoms = False
    with open(run_path) as f:
        for line in f:
            if not in_atoms:
                if line.strip() == "Atoms":
                    in_atoms = True
                continue
            if line.strip() == "End":
                break
            lines.append(line.rstrip("\n"))
    return lines


def write_run(path, basename, atoms_lines, cpl_blocks):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n\n")
        f.write(f"export AMS_JOBNAME={basename}\n")
        f.write("$AMSBIN/ams <<eor\n")
        f.write("System\n  Atoms\n")
        for line in atoms_lines:
            f.write(line + "\n")
        f.write("  End\nEnd\n\n")
        f.write("Task SinglePoint\n\n")
        f.write("Engine ADF\n")
        f.write(f"  title {basename}\n")
        f.write("  NumericalQuality Excellent\n")
        f.write("  Basis\n    Type TZ2P\n    core None\n  End\n")
        f.write("  save TAPE10\n")
        f.write("  symmetry NOSYM\n")
        f.write("  XC\n    GGA PBE\n  End\n")
        f.write("  Relativity\n    Level None\n  End\n")
        f.write("EndEngine\neor\n")
        for block in cpl_blocks:
            resp_str = " ".join(str(r) for r in block['resp'])
            f.write(f"\n# {block['label']}\n")
            f.write("$AMSBIN/cpl << eor\n")
            f.write(f"  adffile {basename}.results/adf.rkf\n")
            f.write(f"  tape10file {basename}.results/TAPE10\n")
            f.write("  nmrcoupling\n")
            f.write(f"    atompert {block['pert']}\n")
            f.write(f"    atomresp {resp_str}\n")
            f.write("  end\neor\n")


def write_sl(path, basename):
    job_id = basename.replace("MDStep", "").replace("_cluster", "")
    with open(path, "w") as f:
        f.write(f"""\
#!/bin/bash

#SBATCH --exclusive
#SBATCH -J ch{job_id}
#SBATCH --output ch{job_id}.o%J
#SBATCH --error  ch{job_id}.e%J
#SBATCH --partition {PARTITION}
#SBATCH --time {WALLTIME}

#SBATCH --ntasks {NTASKS}
#SBATCH --cpus-per-task {CPUS_PER_TASK}

module purge
module load {MODULE}
module list

env | grep SCMLICENSE

CASE={basename}
INP=${{CASE}}.run
OUT=${{CASE}}.out

cp $INP $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR
echo Working directory : $PWD

export SCM_GPUENABLED="FALSE"

set -x
sh $INP > $OUT
set +x

mkdir -p $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
mv * $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
""")


OUT_DIR.mkdir(parents=True, exist_ok=True)

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    basename = os.path.splitext(os.path.basename(xf))[0]

    if SNAPSHOTS is not None:
        step = int(basename.replace("MDStep", "").replace("_cluster", ""))
        if step not in SNAPSHOTS:
            continue

    src_run = SRC_DIR / f"{basename}.run"
    if not src_run.exists():
        print(f"skip {basename}: no TZ2P_FC source run")
        continue

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)

    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    atoms_lines = extract_atoms_block(str(src_run))

    cpl_blocks = []
    for ui, u in enumerate(ureas):
        c_urea = next(at for at in u.atoms if at.symbol == 'C')

        for ci, ch in enumerate(cholines):
            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            if pair is None:
                continue
            _, hN, _, hO = pair

            ch3_hs = [h for _, hs in find_xh_groups(ch, 'C', 3, neighbour_symbol='N')
                      for h in hs]

            resp = hN + hO + ch3_hs
            if DIST_THRESHOLD is not None:
                resp = [h for h in resp if distance(c_urea, h) <= DIST_THRESHOLD]
            if not resp:
                continue

            cpl_blocks.append({
                'label': f"Urea {ui+1} - Choline {ci+1}  [C(urea)-H(CH2,CH3)]",
                'pert':  c_urea.cluster_id,
                'resp':  [h.cluster_id for h in resp],
            })

    if not cpl_blocks:
        print(f"skip {basename}: no CPL pairs")
        continue

    write_run(str(OUT_DIR / f"{basename}.run"), basename, atoms_lines, cpl_blocks)
    write_sl(str(OUT_DIR / f"{basename}.sl"), basename)
    print(f"{basename}: {len(cpl_blocks)} CPL blocks")

finish()
