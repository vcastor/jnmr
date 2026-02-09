#!/usr/bin/python3
import os
import sqlite3
import subprocess

config = {
    "db_path": "nmr_jcoupling.db",
    "variants": [
        {"name": "TZ2P_FC",   "output_dir": "amsoutput/TZ2P_FC"},
        {"name": "TZ2P_all",  "output_dir": "amsoutput/TZ2P_all"},
        {"name": "TZ2PJ_FC",  "output_dir": "amsoutput/TZ2PJ_FC"},
        {"name": "TZ2PJ_all", "output_dir": "amsoutput/TZ2PJ_all"},
    ],
}

conn   = sqlite3.connect(config["db_path"])
cursor = conn.cursor()

for var in config["variants"]:
    variant    = var["name"]
    output_dir = var["output_dir"]
    comment_col = f"comment_{variant}"

    for pattern, message in [
        ("SCF MODERATELY CONVERGED", "SCF MODERATELY CONVERGED"),
        ("SCF NOT CONVERGED",        "SCF NOT CONVERGED"),
    ]:
        result = subprocess.run(
            f"grep -rl '{pattern}' {output_dir}",
            shell=True, capture_output=True, text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            continue

        for filepath in result.stdout.strip().split('\n'):
            filename = filepath.split('/')[-1]
            n_step = int(filename.replace('MDStep', '').replace('_cluster.out', ''))
            cursor.execute(
                f"UPDATE snapshots SET {comment_col} = ? WHERE n_step = ?",
                (message, n_step)
            )

conn.commit()
conn.close()
