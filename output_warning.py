#!/usr/bin/python3
import os
import sqlite3
import subprocess

config = {
    "db_path": "nmr_jcoupling.db",
    "output_dir": "amsoutput",
}

conn   = sqlite3.connect(config["db_path"])
cursor = conn.cursor()

result = subprocess.run(
    f"grep -rl 'SCF MODERATELY CONVERGED' {config['output_dir']}",
    shell=True, capture_output=True, text=True
)

if result.returncode == 0 and result.stdout.strip():
    for filepath in result.stdout.strip().split('\n'):
        # Extract step number from filename like MDStep12345_cluster.out
        filename = filepath.split('/')[-1]
        n_step = int(filename.replace('MDStep', '').replace('_cluster.out', ''))
        cursor.execute(
            "UPDATE snapshots SET comment = 'SCF MODERATELY CONVERGED' WHERE n_step = ?",
            (n_step,)
        )

conn.commit()
conn.close()
