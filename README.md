# Workflow overview

This repository contains snapshots extracted from a Molecular Dynamics (MD) simulation.
Each snapshot is post-processed and used to generate single-point DFT calculation.

## Steps to run

1. **Update the `mdStepsrkf` directory**
   Update the MD snapshot directory using the helper script provided inside `mdStepsrkf`.

2. **Convert RKF files to XYZ**

   ```bash
   $AMSBIN/plams rkf_to_xyz.py
   ```

3. **Select regions**
   Define the regions of interest for each snapshot:

   ```bash
   $AMSBIN/plams region_selector.py
   ```

4. **Generate run files and update the database**
   This step creates all input/run files and updates the database accordingly:

   ```bash
   $AMSBIN/plams run_generator.py
   ```

5. **Transfer files to Rameau**
   Send the generated files to the Rameau server:

   ```bash
   ./to_rameau.sh
   ```

6. **Prepare launchers on Rameau**
   On Rameau, run the provided script to generate the launcher files for each calculation.

7. **Transfer jobs to Criann**
   From Rameau, send the prepared jobs to Criann:

   ```bash
   ./to_criann.sh
   ```

8. **Queue monitoring and execution**

   * On Rameau, `watcher.sh` monitors queue availability on Criann.
   * On Criann, `master.sh` launches the calculations, and a cleaner script manages disk space.

