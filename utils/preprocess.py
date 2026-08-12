"""
utils/preprocess.py
--------------------
Small helper functions shared between train_model.py and app.py, so both
files always agree on how data is encoded and interpreted:

- EDUCATION_MAP        : turns "Bachelor's" etc. into a number the ANN can use.
- count_comma_separated: turns "Python, SQL, Excel" into 3.
- build_feature_vector : assembles the 5 raw inputs in the exact column
                         order the model was trained on.
- scale_features       : applies the SAME StandardScaler used in training.
- get_category         : turns a 0-100 score into Excellent/Good/Average/
                         Needs Improvement.
- extract_text_from_pdf: pulls raw text out of an uploaded resume PDF, for
                         preview only (see note below).
"""

import numpy as np
import joblib

# Ordinal mapping for Education Level. Higher = more advanced degree.
# Keeping this in ONE place means train_model.py and app.py can never
# accidentally disagree on how education is encoded.
EDUCATION_MAP = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor's": 3,
    "Master's": 4,
    "PhD": 5,
}

# The exact order of columns the model was trained on.
FEATURE_COLUMNS = [
    "Years_of_Experience",
    "Education_Level_Encoded",
    "Number_of_Skills",
    "Certifications",
    "Projects",
]


def encode_education(level_str):
    """Convert an education level string into its ordinal numeric code."""
    return EDUCATION_MAP.get(level_str, EDUCATION_MAP["Bachelor's"])


def count_comma_separated(text):
    """
    Count non-empty comma-separated items in a string.
    e.g. "Python, SQL, , Excel" -> 3
    Used so the Streamlit form can accept a natural "Python, SQL, Excel"
    text entry while the model still just sees a simple count.
    """
    if not text:
        return 0
    return len([item.strip() for item in text.split(",") if item.strip()])


def build_feature_vector(years_experience, education_level, num_skills, certifications, projects):
    """Assemble the 5 raw features into a single row, in the model's expected order."""
    education_encoded = encode_education(education_level)
    return np.array(
        [[years_experience, education_encoded, num_skills, certifications, projects]],
        dtype=float,
    )


def scale_features(features, scaler_path="scaler.pkl"):
    """Apply the same StandardScaler that was fit during training."""
    scaler = joblib.load(scaler_path)
    return scaler.transform(features)


def get_category(score):
    """
    Map a predicted 0-100 score to a human-readable screening category.

    Thresholds are calibrated against the actual distribution of scores
    produced by the trained model (see train_model.py / dataset/resumes.csv)
    rather than arbitrary round numbers, so the categories stay meaningful:
    roughly the top 10% of candidates land in "Excellent", the next 15%
    in "Good", the middle third in "Average", and the rest in
    "Needs Improvement".
    """
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 35:
        return "Average"
    else:
        return "Needs Improvement"


def extract_text_from_pdf(uploaded_file):
    """
    Extract raw text from an uploaded PDF, for preview purposes ONLY.

    Important: this text is never fed into the model. The predicted score
    comes entirely from the structured fields (experience, education,
    skills, certifications, projects). This keeps the project a simple,
    explainable ANN instead of an NLP/LLM system.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else "(No selectable text found in this PDF.)"
    except Exception as exc:
        return f"(Could not extract a text preview: {exc})"
