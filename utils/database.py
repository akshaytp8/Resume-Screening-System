"""
utils/database.py
------------------
Tiny SQLite helper module. Every time someone screens a candidate in
app.py, their inputs + predicted score are saved here so they show up
in the "Screening History" tab. Uses only Python's built-in sqlite3
module - no extra database server needed.
"""

import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "database.db"
TABLE_NAME = "screening_history"


def get_connection():
    """Open a connection to the local SQLite file (created if missing)."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the screening_history table if it doesn't exist yet.
    Safe to call every time the app starts."""
    conn = get_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name   TEXT,
            resume_filename  TEXT,
            years_experience REAL,
            education_level  TEXT,
            num_skills       INTEGER,
            certifications   INTEGER,
            projects         INTEGER,
            predicted_score  REAL,
            category         TEXT,
            created_at       TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(candidate_name, resume_filename, years_experience, education_level,
                       num_skills, certifications, projects, predicted_score, category):
    """Insert one screening result as a new row."""
    conn = get_connection()
    conn.execute(f"""
        INSERT INTO {TABLE_NAME} (
            candidate_name, resume_filename, years_experience, education_level,
            num_skills, certifications, projects, predicted_score, category, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        candidate_name, resume_filename, years_experience, education_level,
        num_skills, certifications, projects, predicted_score, category,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def fetch_all_predictions():
    """Return every past screening as a pandas DataFrame, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC", conn)
    conn.close()
    return df
