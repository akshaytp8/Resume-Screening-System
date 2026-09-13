import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "database.db"
TABLE_NAME = "screening_history"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
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
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC", conn)
    conn.close()
    return df
