"""
app.py
------
Streamlit frontend for the Intelligent Resume Screening System.

What it does:
1. Collects candidate details (experience, education, skills,
   certifications, projects) plus an optional resume PDF.
2. Feeds the structured features into the trained ANN (model.h5) to
   predict a Resume Score out of 100.
3. Buckets the score into Excellent / Good / Average / Needs Improvement.
4. Saves every screening to a local SQLite database and shows past
   results in a "Screening History" tab.

Run:
    streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow import keras

from utils.database import init_db, insert_prediction, fetch_all_predictions
from utils.preprocess import (
    EDUCATION_MAP,
    get_category,
    extract_text_from_pdf,
    count_comma_separated,
)

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.json"
UPLOAD_DIR = "uploaded_resumes"

CATEGORY_STYLE = {
    "Excellent": ("🟢", "#0F766E"),
    "Good": ("🔵", "#2563EB"),
    "Average": ("🟡", "#CA8A04"),
    "Needs Improvement": ("🔴", "#DC2626"),
}

st.set_page_config(
    page_title="Intelligent Resume Screening System",
    page_icon="🧠",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached resources - loaded once per session instead of on every rerun
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    # compile=False: the app only runs inference (model.predict), never
    # trains further, so we don't need the optimizer/loss state restored.
    # This also sidesteps a Keras HDF5-loading bug where the saved "mse"
    # metric config fails to deserialize on load.
    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)
model, scaler = load_model_and_scaler()

# Streamlit reruns the whole script on every interaction (every click,
# every keystroke in some widgets). We stash the latest prediction in
# session_state so the result panel doesn't disappear on the next rerun.
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Scores a candidate's resume (0-100) using a simple feed-forward "
        "Artificial Neural Network trained on structured candidate "
        "features - no NLP or LLMs involved."
    )

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        st.subheader("Model performance (test set)")
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{metrics['mae']:.2f}")
        c2.metric("RMSE", f"{metrics['rmse']:.2f}")
        st.caption(f"R² score: {metrics.get('r2', 0):.3f}")

    st.subheader("Score categories")
    st.markdown(
        "- 🟢 **Excellent** — 80 to 100\n"
        "- 🔵 **Good** — 60 to 79\n"
        "- 🟡 **Average** — 35 to 59\n"
        "- 🔴 **Needs Improvement** — 0 to 34"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("🧠 Intelligent Resume Screening System")
st.caption("ANN-powered resume scoring, built with TensorFlow/Keras & Streamlit")

if model is None:
    st.error(
        "No trained model found. Run `python train_model.py` first to "
        "generate `model.h5` and `scaler.pkl`, then restart this app."
    )
    st.stop()

tab_screen, tab_history = st.tabs(["🔍 Screen a Candidate", "📊 Screening History"])

# =====================================================================
# TAB 1 - Screen a candidate
# =====================================================================
with tab_screen:
    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.subheader("Candidate details")

        candidate_name = st.text_input("Candidate Name", placeholder="e.g., Priya Sharma")

        resume_file = st.file_uploader(
            "Upload Resume (PDF) — optional, stored for reference only",
            type=["pdf"],
            help="The PDF itself is not used for scoring. Scoring uses the fields below.",
        )

        c1, c2 = st.columns(2)
        with c1:
            years_experience = st.number_input(
                "Years of Experience", min_value=0.0, max_value=25.0, value=1.0, step=0.5
            )
        with c2:
            education_level = st.selectbox(
                "Education Level", list(EDUCATION_MAP.keys()), index=2
            )

        skills_input = st.text_input(
            "Skills (comma-separated)", placeholder="Python, SQL, Excel, Communication"
        )
        certifications_input = st.text_input(
            "Certifications (comma-separated)", placeholder="AWS Cloud Practitioner, PMP"
        )
        projects_input = st.text_input(
            "Projects (comma-separated titles)", placeholder="E-commerce site, Chatbot"
        )

        num_skills = count_comma_separated(skills_input)
        certifications = count_comma_separated(certifications_input)
        projects = count_comma_separated(projects_input)

        st.caption(
            f"Detected **{num_skills}** skill(s) · **{certifications}** "
            f"certification(s) · **{projects}** project(s)"
        )

        predict_clicked = st.button(
            "Predict Resume Score", type="primary", width="stretch"
        )

    with col_result:
        st.subheader("Result")

        if predict_clicked:
            features = np.array([[
                years_experience,
                EDUCATION_MAP[education_level],
                num_skills,
                certifications,
                projects,
            ]], dtype=float)

            features_scaled = scaler.transform(features)
            raw_score = float(model.predict(features_scaled, verbose=0).flatten()[0])
            score = float(np.clip(raw_score, 0, 100))
            category = get_category(score)

            # Extract a text preview and save the PDF to disk, if one was uploaded
            preview_text = None
            resume_filename = "N/A"
            if resume_file is not None:
                preview_text = extract_text_from_pdf(resume_file)
                resume_file.seek(0)
                resume_filename = (
                    f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{resume_file.name}"
                )
                with open(os.path.join(UPLOAD_DIR, resume_filename), "wb") as f:
                    f.write(resume_file.getbuffer())

            insert_prediction(
                candidate_name=candidate_name.strip() or "Unnamed Candidate",
                resume_filename=resume_filename,
                years_experience=years_experience,
                education_level=education_level,
                num_skills=num_skills,
                certifications=certifications,
                projects=projects,
                predicted_score=round(score, 2),
                category=category,
            )

            st.session_state.last_result = {
                "score": score,
                "category": category,
                "preview_text": preview_text,
            }

        result = st.session_state.last_result
        if result:
            emoji, color = CATEGORY_STYLE.get(result["category"], ("", "#000000"))
            st.metric("Predicted Resume Score", f"{result['score']:.2f} / 100")
            st.markdown(
                f"<h3 style='color:{color};'>{emoji} {result['category']}</h3>",
                unsafe_allow_html=True,
            )
            st.progress(result["score"] / 100)

            if result["preview_text"]:
                with st.expander("📄 Extracted resume text (preview only, not used in scoring)"):
                    text = result["preview_text"]
                    st.text(text[:1500] + ("..." if len(text) > 1500 else ""))

            st.success("Saved to screening history.")
        else:
            st.info("Fill in the candidate details and click **Predict Resume Score**.")

# =====================================================================
# TAB 2 - History
# =====================================================================
with tab_history:
    st.subheader("Previous screenings")
    history_df = fetch_all_predictions()

    if history_df.empty:
        st.info("No screenings yet - predictions will appear here after you screen a candidate.")
    else:
        st.dataframe(history_df, width="stretch", hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="screening_history.csv",
            mime="text/csv",
        )
