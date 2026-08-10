# DES NMR J-coupling pipeline

Snapshots from a Molecular Dynamics simulation of a urea / choline chloride
Deep Eutectic Solvent are post-processed into small clusters and fed to
single-point DFT calculations to compute NMR J-couplings.  Geometric and
quantum analyses are stored alongside the J values in a SQLite DB and plotted
from there.

## Requirements

- AMS with PLAMS (`$AMSBIN/plams`)
- Python 3 with: `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`,
  `sqlite3` (stdlib)
- A LaTeX install for the `usetex` plots (`xfrac`, `amsmath`). Plot styling
  (font sizes, `usetex`, LaTeX preamble) is centralised in
  `hassan_functions/style.py` via `apply_style(preset)`; the `notex` preset
  renders without LaTeX.

## Directory layout

```
mdSteps/rkf/            # raw MD checkpoints (.rkf), input
mdSteps/xyz/            # xyz dump per MD step
clusters/               # inner-sphere clusters per step
amsoutput/<variant>/    # ADF NMR output, one dir per basis/variant
amsoutput/qtaim/        # ADF QTAIM output
amsoutput/cdft/         # ADF CDFT (conceptual DFT) output
amsoutput/ch/           # ADF C(urea)-H coupling output
amsoutput/nh,nh_intra/  # ADF N(urea)-choline / intra-urea N-H coupling output
amsoutput/site/         # ADF named-site (H1-H5, Nurea) coupling output
run_scripts/<variant>/  # generated .run / .sl scripts
run_scripts/qtaim,cdft/ # generated QTAIM / CDFT scripts
run_scripts/ch/         # generated C(urea)-H coupling scripts (small clusters)
run_scripts/nh,nh_intra/# generated N(urea) coupling scripts
run_scripts/site/       # generated named-site coupling scripts
pipeline/               # generation scripts (rkf -> xyz -> clusters -> run/.sl + DB)
analysis/               # analysis / plotting scripts
analysis/cache/         # pickled PLAMS-compute results (regenerated on data change)
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
./reader.py     # parse outputs into the DB (silent): SCF warnings + J(HH) + QTAIM (DI) /
                # CDFT (chi) + C(urea)-H + N(urea) + named-site couplings (each carrying a
                # per-step SCF comment flag)
./analysis.sh   # run every analysis (plots + reports); regenerates a PLAMS cache only when
                # its clusters/ input or its script changed, otherwise it just re-plots
```

`pipeline/` (generation, run by `generate.sh`):

```
rkf_to_xyz.py          # rkf -> xyz dump (mdSteps/rkf -> mdSteps/xyz)
region_selector.py     # carve inner-sphere clusters at the target ratio
coupling_generator.py  # classify cluster, seed DB rows, write .run + .sl for the 4
                       # NMR variants (skipping SCF-warned / already-run steps).
                       # Opt-in extensions, each generating ONLY that set of scripts:
                       #   CH=1        C(urea)-H       (all choline H within 5 A)
                       #   NH=1        N(urea)-H1
                       #   NH_INTRA=1  N-H within one urea
                       #   SITE=1      the nine named-site couplings (H1-H5, Nurea)
                       # e.g. SITE=1 SITE_LIMIT=10 $AMSBIN/plams pipeline/coupling_generator.py
populate_geometry.py   # backfill intra dihedral / distance / angle columns
property_generator.py  # write QTAIM + CDFT .run/.sl files for converged steps
```

### Submission restrictions (temporary)

Two limits in `pipeline/coupling_generator.py` cap what gets sent to CRIANN. They
apply to **every** branch — the main H-H workflow and all the opt-in extensions — so
a restricted batch stays restricted. Both are meant to be lifted; the constants sit
at the top of the file.

| Constant | Default | Effect | Lift with |
|---|---|---|---|
| `ALLOWED_COMPOSITIONS` | `[(2,1,1)]`  | only clusters of 2 urea + 1 choline + 1 chloride (38 atoms) | `ALLOW_MEDIUM=1` also allows (4,2,2) = 76 atoms; `NO_SIZE_LIMIT=1` allows any; or set the constant to `None` |
| `MIN_STEP`             | `60_000_000` | skip clusters before this MD step, so a new batch does not inherit the sampling window the earlier clusters came from | `MIN_STEP=0`, or edit the constant |

Composition is read from the xyz element tally (`cluster_composition`), not from the
atom count, so an off-ratio carve such as (5,2,2) is rejected rather than silently
counted as a near-76-atom cluster. Currently available under the defaults: 25
clusters at (2,1,1), or 66 with `ALLOW_MEDIUM=1`.

`analysis/` (run by `analysis.sh`). The PLAMS-compute scripts read
`clusters/*.xyz` and dump `analysis/cache/*.pkl`; the plotters read those caches
and/or the DB, so tweaking a plot never re-reads every cluster (the compute step
is skipped unless its input changed):

```
# PLAMS compute — need $AMSBIN/plams, write analysis/cache/*.pkl
distance_intra.py      # intra C-H / H-H distances + H-C-C-H, N-CH2-CH2-O dihedrals
distance_inter.py      # inter urea-choline distances, incl. double bridges
gauche_anti.py         # N-CH2-CH2-O gauche vs anti populations
qtaim_analysis.py      # BCP rho/Gb/Vb + QTAIM net charges + Poincare-Hopf count
choline_fold.py        # is the choline folded? intramolecular O...H(CH3) BCP + ring CP

# plotters / reports — plain python, read caches and/or the DB
distance_plot.py       # intra + inter distance / dihedral plots (merged intra+inter)
gauche_anti_plot.py    # gauche vs anti histogram
qtaim_analysis_plot.py # BCP properties vs distance + net-charge histogram
visualiser_data.py     # |J| (HH) distributions per basis/variant; cubic-mean effective J
ch_coupling.py         # |J| C(urea)-H couplings; cubic-mean box, SCF-flagged rows excluded
karplus_fit.py         # Karplus + descriptor fits, RMSE summary

# standalone — not run by analysis.sh
direct_dij.py          # direct dipolar coupling |D_ij| histogram
karplus_ml.py          # ML model — memory-heavy, run manually
```

All J stats/plots use `|J|` (the sign is meaningless vs experiment); the
representative average is the "cubic" power mean (`hassan_functions/jstats.py`).

## Named-site J couplings

The experimental team labels the NMR sites H1-H5 and Nurea. These are defined once,
machine-readably, in `hassan_functions/constants.SITE_LABELS`; the table below is the
same thing for humans. Choline is (CH3)3N(+)-CH2-CH2-OH, urea is H2N-CO-NH2.

| Label | Species | Group | Per molecule | Meaning |
|---|---|---|---|---|
| `H1`    | choline | CH3            | 9 | H on a methyl carbon of the trimethylammonium head |
| `H2`    | choline | CH2 next to N+ | 2 | H on the CH2 bonded to the quaternary N |
| `H3`    | choline | CH2 next to O  | 2 | H on the CH2 bonded to the hydroxyl O |
| `H4`    | choline | OH             | 1 | hydroxyl H |
| `H5`    | urea    | NH2            | 4 | amine H — the only H in urea |
| `Nurea` | urea    | NH2            | 2 | amide N of urea |

The couplings requested by the experimental team, all stored in the single
`site_coupling` table and told apart by its `pair_type` and `scope` columns
(`hassan_functions/constants.SITE_COUPLINGS` is the machine-readable copy):

| `pair_type` | `scope` | What it is |
|---|---|---|
| `H1-H2`    | `intra`          | methyl H to the CH2 next to N+, same choline |
| `H1-H3`    | `intra`, `inter` | methyl H to the CH2 next to O — within one choline, and between two |
| `H1-H4`    | `intra`, `inter` | methyl H to the hydroxyl H — within one choline, and between two |
| `H5-H2`    | `inter`          | urea amine H to the choline CH2 next to N+ |
| `H5-H3`    | `inter`          | urea amine H to the choline CH2 next to O |
| `H5-H4`    | `inter`          | urea amine H to the choline hydroxyl H |
| `Nurea-H1` | `inter`          | urea N to a choline methyl H |
| `Nurea-H2` | `inter`          | urea N to the choline CH2 next to N+ |
| `Nurea-H3` | `inter`          | urea N to the choline CH2 next to O |

`scope` is `intra` when both atoms belong to the same molecule and `inter` when they
do not; a pair whose two sites sit on different species can only ever be `inter`. A
row also carries `pert_site` / `resp_site` (which label each atom actually got), so
the assignment is auditable without recomputing it.

Every J table in the DB, and what it holds:

| Table | Coupling | Perturber → responder |
|---|---|---|
| `step_<n>_intra`    | H-H, CH2-CH2 within a choline      | choline H → choline H |
| `step_<n>_inter`    | H-H, through-space NH2···CH3       | urea H → choline methyl H |
| `ch_coupling`       | C(urea)-H(choline)                 | urea C → choline H |
| `nh_coupling`       | N(urea)-choline                    | urea N → choline H / CH2 C |
| `nh_intra_coupling` | N-H within one urea                | urea N → its own H |
| `site_coupling`     | the nine named-site couplings above | see `pair_type` |

Generating them: one cpl block covers several requested couplings at once (cpl takes
one perturber against many responders), so the perturber is always the sparser site
and `reader.classify_site` re-derives each pair's `pair_type` from the geometry. A
2:1 cluster needs only ~9-15 blocks for all nine couplings.

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
| `sites`     | `choline_sites`, `urea_sites`, `cluster_sites` — locate the named NMR sites (H1-H5, Nurea) in a molecule |
| `cache`     | `save_cache(name, data)`, `load_cache(name)` — pickle PLAMS-compute results under `analysis/cache/` |
| `criann`    | all CRIANN/SLURM config: `slurm_script(jobname, case, partition, ...)` + the shared `#SBATCH` template, `PARTITION_WALLTIME` (court, tcourt, long, tlong), `VARIANT_SLURM` (per-variant partition/walltime), `SCF_WARNINGS`, `MODULE`, `NTASKS`, `CPUS_PER_TASK` |
| `plotting`  | `PLOT_STYLES`, `hist`, `mlabel`, `stats`, `style_axes`, `style_cbar`, `save_fig` (opaque→PDF, transparent→SVG) |
| `style`     | `apply_style(preset)` — centralised matplotlib rcParams (font sizes, `usetex`, LaTeX preamble); presets `default` / `large` / `notex` |
| `jstats`    | `cubic_mean`, `cubic_dispersion`, `effective_n`, `cubic_mean_ci` — the p=3 power-mean "effective J" used for every J coupling (HH + CH), with snapshot-level error bars |
| `constants` | `FORMULAS` (urea, choline, chloride, water), `SITE_LABELS` / `SITE_COUPLINGS` / `pair_type` (the named-site dictionary above), NMR `ISOTOPE_FOR_SYMBOL` and `GAMMA`, physical constants (`MU0_OVER_4PI`, `HBAR`, `TWO_PI`, `ANGSTROM_TO_M`) |

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
