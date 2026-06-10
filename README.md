# DES NMR J-coupling pipeline

Snapshots from a Molecular Dynamics simulation of a urea / choline chloride
Deep Eutectic Solvent are post-processed into small clusters and fed to
single-point DFT calculations to compute NMR J-couplings.  Geometric and
qunatum analyses are stored alongside the J values in a SQLite DB and plotted
from there.

## Requirements

- AMS with PLAMS (`$AMSBIN/plams`)
- Python 3 with: `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`,
  `sqlite3` (stdlib)
- A LaTeX install for the publication-quality plots
  (`text.usetex = True`, `xfrac` package)

## Directory layout

```
mdSteps/rkf/            # raw MD checkpoints (.rkf), input
mdSteps/xyz/            # xyz dump per MD step
clusters/               # inner-sphere clusters per step
amsoutput/<variant>/    # ADF NMR output, one dir per basis/variant
amsoutput/qtaim/        # ADF QTAIM output
amsoutput/cdft/         # ADF CDFT (conceptual DFT) output
amsoutput/ch/           # ADF C(urea)-H coupling output
run_scripts/<variant>/  # generated .run / .sl scripts
run_scripts/qtaim,cdft/ # generated QTAIM / CDFT scripts
run_scripts/ch/         # generated C(urea)-H coupling scripts (small clusters)
pipeline/               # generation scripts (rkf -> xyz -> clusters -> run/.sl + DB)
analysis/               # analysis / plotting scripts
plots/                  # output figures
nmr_jcoupling.db        # SQLite database (see init_db.py for schema)
hassan_functions/       # shared library, see below
```

## Running

All commands are run from the repo root.

```
init_db.py      # one-shot: create the DB schema (run once)
./generate.sh   # download new .rkf from aragorn, then run pipeline/ (below)
./criann.sh     # upload .run/.sl, download outputs, then clean + submit on CRIANN
./reader.py     # parse outputs into the DB: SCF warnings + J + QTAIM (DI) / CDFT (chi)
./analysis.sh   # run every analysis (plots + reports)
```

`pipeline/` (generation, run by `generate.sh`):

```
rkf_to_xyz.py          # rkf -> xyz dump (mdSteps/rkf -> mdSteps/xyz)
region_selector.py     # carve inner-sphere clusters at the target ratio
coupling_generator.py  # classify cluster, seed DB rows, write .run + .sl for the 4
                       # NMR variants (skipping SCF-warned / already-run steps).
                       # Opt-in extension: CH=1 $AMSBIN/plams pipeline/coupling_generator.py
                       # generates ONLY the C(urea)-H scripts (court, all choline H within 5 A)
populate_geometry.py   # backfill intra dihedral / distance / angle columns
property_generator.py  # write QTAIM + CDFT .run/.sl files for converged steps
```

`analysis/` (run by `analysis.sh`):

```
distance_intra.py      # intra distance / dihedral plots
distance_inter.py      # inter distance plots, incl. double bridges
gauche_anti.py         # gauche vs anti populations
qtaim_analysis.py      # BCP rho/Gb/Vb vs distance + QTAIM net-charge histogram +
                       # Poincare-Hopf satisfied count
visualiser_data.py     # |J| distributions per basis / variant
karplus_fit.py         # Karplus + distance fits, RMSE summary
direct_dij.py          # direct dipolar coupling |D_ij| histogram
karplus_ml.py          # ML model — memory-heavy, run manually (not from analysis.sh)
```

## Atom indexing

Atom numbers stored in the DB (`H_pert`, `H_resp`) and used by ADF / QTAIM /
CDFT input and output refer to a **canonical reordering** of the cluster,
NOT the order atoms have in `clusters/*.xyz`. The reordering is the one
`pipeline/coupling_generator.py` applies when it writes the `.run` files:

1. `cluster.separate()` splits the cluster into molecules.
2. `classify_sort(mols, centre, FORMULAS)` groups by species, BFS-canonicalises
   atoms inside each molecule, and sorts each species list by COM-distance to
   the cluster centre.
3. Atoms are concatenated in the `SPECIES` order
   (`['urea', 'choline', 'chloride']`) and numbered globally from 1.

Any script that needs to look up a value by atom index (charge, DI, chi,
BCP) **must** reproduce this reordering and tag each atom with the global
index before querying:

```python
mol_data = classify_sort(cluster.separate(), centre,
                         {k: FORMULAS[k] for k in SPECIES})
offs = compute_offsets(mol_data, SPECIES)
for name in SPECIES:
    for mi, mol in enumerate(mol_data[name]):
        off = offs[name][mi]
        for ai, at in enumerate(mol.atoms):
            at.cluster_id = off + ai + 1
```

Scripts that work only with `at.coords` (e.g. `analysis/distance_intra.py`,
`analysis/distance_inter.py`) don't need this — they never cross the
xyz ↔ ADF-output index boundary.

## `hassan_functions/` library

Shared utilities split by topic. All molecule-specific information is
held in `constants.FORMULAS`, so the pipeline can be retargeted to a
different DES (e.g. ratio 1:3, urea swapped for water) by editing the
formula dictionary and the species list in the few scripts that
classify by species.

| Module | Contents |
|---|---|
| `geometry`  | `distance(a, b)`, `dihedral(a, b, c, d)` |
| `finders`   | `find_atoms`, `find_xh_groups`, `find_xh_bonds`, `find_adjacent_xh_pairs`, `find_adjacent_xh_pair_anchored` — generic finders driven by atom symbol and H count, with optional `neighbour` filter and `return_indices` flag |
| `ordering`  | `canonical_order`, `reorder`, `classify_sort`, `compute_offsets` — BFS canonicalisation of atom indices and species-based sorting of clusters |
| `io`        | `get_step_from_filename`, `read_xyz`, `normalise_symbol`, `read_labeled_matrix`, `read_qtaim_charges` |
| `db`        | `table_exists`, `column_exists` (SQLite helpers) |
| `criann`    | all CRIANN/SLURM config: `slurm_script(jobname, case, partition, ...)` + the shared `#SBATCH` template, `PARTITION_WALLTIME` (court, tcourt, long, tlong), `VARIANT_SLURM` (per-variant partition/walltime), `SCF_WARNINGS`, `MODULE`, `NTASKS`, `CPUS_PER_TASK` |
| `plotting`  | `PLOT_STYLES`, `hist`, `mlabel`, `stats`, `style_axes`, `style_cbar` |
| `constants` | `FORMULAS` (urea, choline, chloride, water), NMR `ISOTOPE_FOR_SYMBOL` and `GAMMA`, physical constants (`MU0_OVER_4PI`, `HBAR`, `TWO_PI`, `ANGSTROM_TO_M`) |

### Adapting to a different system

1. Add or change a formula in `hassan_functions/constants.FORMULAS`.
2. Update the `SPECIES` list in `pipeline/coupling_generator.py` and
   `pipeline/populate_geometry.py` if the species set changes.
3. Replace finder calls with the relevant symbols, e.g.
   `find_xh_groups(water, 'O', 2)` instead of
   `find_xh_groups(urea, 'N', 2)`.
4. Change the target ratio in `pipeline/region_selector.py`.

## CRIANN helpers

`criann.sh` does the whole round-trip: it calls the two rsync helpers, then
pipes its embedded clean + submit block to CRIANN over ssh.

```
run_scripts/to_criann.sh      # rsync .run/.sl up to CRIANN
amsoutput/download_outputs.sh # rsync .out files back down
```
