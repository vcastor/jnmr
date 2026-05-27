# DES NMR J-coupling pipeline

Snapshots from a Molecular Dynamics (MD) simulation of a urea / choline
chloride Deep Eutectic Solvent are post-processed into small clusters and
fed to single-point DFT calculations to compute proton NMR J-couplings.
Geometric and topological analyses (distances, dihedrals, QTAIM BCPs,
delocalisation indices) are stored alongside the J values in a SQLite DB
and plotted from there.

## Requirements

- AMS / ADF with PLAMS (`$AMSBIN/plams`)
- Python 3 with: `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`,
  `sqlite3` (stdlib)
- A LaTeX install for the publication-quality plots
  (`text.usetex = True`, `xfrac` package)

## Directory layout

```
mdStepsrkf/                   # raw MD checkpoints (.rkf), input
mdStepsxyz/                   # xyz dump per MD step
clusters/                     # inner-sphere clusters per step
amsoutput/<variant>/          # ADF NMR output, one dir per basis/variant
amsoutput/qtaim/              # ADF QTAIM output
run_scripts/<variant>/        # generated .run / .sl scripts
plots/                        # output figures
nmr_jcoupling.db              # SQLite database (see init_db.py for schema)
hassan_functions/             # shared library, see below
```

## Pipeline order

```
rkf_to_xyz.py        # rkf -> xyz dump (mdStepsrkf -> mdStepsxyz)
region_selector.py   # carve inner-sphere clusters at the target ratio
init_db.py           # one-shot DB schema setup
run_generator.py     # classify cluster, write ADF .run files, seed DB rows
                     # (submit on cluster: run_criann.sh)
output_warning.py    # mark SCF-not-converged steps in the DB
output_reader.py     # parse ADF output, fill J columns
populate_intra_dihedral.py
                     # backfill intra dihedral / distance columns
qtaim_generator.py   # write QTAIM .run files for the converged steps
qtaim_reader.py      # read DI matrices into the DB
```

Analysis (run after the data is in):

```
distance_intra.py            # intra distance / dihedral plots
distance_inter.py            # inter distance plots, incl. double bridges
qtaim_distance_analysis.py   # BCP rho vs distance scatter / histogram
visualiser_data.py           # |J| distributions per basis / variant
karplus_fit.py               # Karplus + distance fits, RMSE summary
direct_dij.py                # direct dipolar coupling |D_ij| histogram
regression_dij.py            # DI vs |J| scatter
find_missing_tz2p_fc.py      # report steps missing a variant
```

## `hassan_functions/` library

Shared utilities split by topic. All molecule-specific information is
held in `constants.FORMULAS`, so the pipeline can be retargeted to a
different DES (e.g. ratio 1:3, urea swapped for water) by editing the
formula dictionary and the species list in the few scripts that
classify by species.

| Module | Contents |
|---|---|
| `geometry`  | `distance(a, b)`, `dihedral(a, b, c, d)` |
| `finders`   | `find_atoms`, `find_xh_groups`, `find_xh_bonds`, `find_adjacent_xh_pairs`, `find_adjacent_xh_pair_anchored` — generic finders driven by atom symbol and H count, with optional `neighbour_symbol` filter and `return_indices` flag |
| `ordering`  | `canonical_order`, `reorder`, `classify_sort`, `compute_offsets` — BFS canonicalisation of atom indices and species-based sorting of clusters |
| `io`        | `get_step_from_filename`, `read_xyz`, `normalise_symbol` |
| `db`        | `table_exists`, `column_exists` (SQLite helpers) |
| `plotting`  | `PLOT_STYLES`, `hist`, `mlabel`, `stats`, `style_axes`, `style_cbar` |
| `constants` | `FORMULAS` (urea, choline, chloride, water), NMR `ISOTOPE_FOR_SYMBOL` and `GAMMA`, physical constants (`MU0_OVER_4PI`, `HBAR`, `TWO_PI`, `ANGSTROM_TO_M`) |

### Adapting to a different system

1. Add or change a formula in `hassan_functions/constants.FORMULAS`.
2. Update the `SPECIES` list in `run_generator.py` and
   `populate_intra_dihedral.py` if the species set changes.
3. Replace finder calls with the relevant symbols, e.g.
   `find_xh_groups(water, 'O', 2)` instead of
   `find_xh_groups(urea, 'N', 2)`.
4. Change the target ratio in `region_selector.py`.

## Cluster submit / housekeeping

```
run_criann.sh         # submit jobs on the CRIANN partition
clean_criann.sh       # tidy old jobs / temp dirs
update.sh             # pull results from the cluster
```
