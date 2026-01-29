#!/usr/bin/python3
import os
import sys
import sqlite3

def init_database(db_path: str) -> None:
    """Initialize the NMR J-coupling database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main table with snapshot info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            n_step INTEGER PRIMARY KEY,
            n_choline INTEGER,
            n_inter INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    
def add_comment_column(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        ALTER TABLE snapshots
        ADD COLUMN comment TEXT
    """)
    conn.commit()
    conn.close()

# Main execution

db_path = "nmr_jcoupling.db"
# init_database(db_path)
add_comment_column(db_path)

