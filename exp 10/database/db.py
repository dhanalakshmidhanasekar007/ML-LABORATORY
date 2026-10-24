"""
SQLite Database Helper for Logging Skin Disease Predictions
Database file: database/skin_disease.db
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "skin_disease.db")


def init_db():
    """Initializes the SQLite prediction history database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(filename, prediction, confidence):
    """Logs an inference result into the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created_at = datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p")
    cur.execute("""
        INSERT INTO history (filename, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (filename, prediction, float(confidence), created_at))
    conn.commit()
    conn.close()


def get_history(limit=25):
    """Retrieves recent prediction records."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def clear_history():
    """Clears all logged history records."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
