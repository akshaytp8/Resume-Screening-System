# 🧠 Intelligent Resume Screening System

A simple ANN-based resume screening tool built with **TensorFlow/Keras**, **Streamlit**, and **SQLite**. Enter a candidate's experience, education, skills, certifications, and projects, and the model predicts a **Resume Score (0–100)** and buckets it into Excellent / Good / Average / Needs Improvement.

Built as a portfolio / academic project (MCA-level) to demonstrate an end-to-end ML workflow: synthetic data generation → preprocessing → a small feed-forward neural network → evaluation → a working web front end → persistent storage.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project is **intentionally simple**: no NLP, no transformers, no LLM APIs. The "resume" is represented by five structured numbers, and a small ANN learns to map those numbers to a score. That keeps the deep-learning concepts (architecture, loss, optimizer, evaluation metrics, feature scaling) front and center, which is the point of the exercise.

A resume PDF *can* be uploaded through the UI, but it's stored for reference only and previewed as extracted text — it does **not** feed into the model.

> 📸 **Add a screenshot here** — run the app locally (see below), take a screenshot of the "Screen a Candidate" tab, and drop it in as `screenshot.png` for your GitHub README.

## Features

- **Manual candidate entry** — years of experience, education level, comma-separated skills/certifications/projects (auto-counted).
- **Optional resume PDF upload** — stored on disk, with an extracted-text preview.
- **ANN-powered prediction** — a 0–100 Resume Score from a trained Keras model.
- **Category bucketing** — Excellent / Good / Average / Needs Improvement.
- **Persistent history** — every screening is saved to SQLite and browsable in-app.
- **CSV export** — download the full screening history.
- **Model performance in the sidebar** — live MAE / RMSE / R² from the last training run.

## Tech stack

| Layer                | Tool                          |
|-----------------------|--------------------------------|
| Model                 | TensorFlow / Keras (ANN)      |
| Data processing        | Pandas, NumPy                 |
| Train/test split & metrics | scikit-learn              |
| Frontend               | Streamlit                     |
| Storage                | SQLite                        |
| PDF text preview        | pypdf                         |
| Version control          | Git / GitHub                 |

## Project structure

```
Resume-Screening-System/
│
├── app.py                  # Streamlit web app (the UI you interact with)
├── train_model.py          # Generates the dataset (if needed), trains & saves the ANN
├── model.h5                # Trained Keras model (produced by train_model.py)
├── scaler.pkl               # Fitted StandardScaler, needed to preprocess new inputs the same way
├── metrics.json              # MAE / RMSE / R² from the last training run (shown in the app sidebar)
├── training_history.png       # Train vs. validation loss curve
├── requirements.txt         # Python dependencies
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml          # App theme
├── dataset/
│   └── resumes.csv          # Synthetic training data (3,000 rows)
├── utils/
│   ├── __init__.py
│   ├── database.py          # SQLite helpers: init_db, insert_prediction, fetch_all_predictions
│   └── preprocess.py        # Shared encoding/scaling/category logic used by both app.py and train_model.py
└── README.md
```

`database.db` and `uploaded_resumes/` are **not** committed — they're created automatically the first time you run the app (see `.gitignore`). This keeps the repo free of local runtime data while still being fully reproducible.

### What each file does

- **`app.py`** — the Streamlit frontend. Loads `model.h5` and `scaler.pkl`, renders the input form and history table, and writes each prediction to SQLite via `utils/database.py`.
- **`train_model.py`** — a standalone script. If `dataset/resumes.csv` doesn't exist, it generates one; then it preprocesses the data, trains the ANN, evaluates it, and saves `model.h5`, `scaler.pkl`, `metrics.json`, and `training_history.png`. Run this whenever you want to retrain.
- **`utils/preprocess.py`** — the single source of truth for how raw inputs become model-ready numbers (education encoding, feature ordering, scaling) and how a score maps to a category. Both `app.py` and `train_model.py` import from here so the two can never drift out of sync.
- **`utils/database.py`** — small SQLite wrapper: create the table, insert a row, fetch history as a DataFrame.

## Dataset

No public "resume score" dataset really exists, so `train_model.py` **generates a realistic synthetic dataset** (3,000 rows by default) the first time it runs:

| Column                | Description                                  |
|------------------------|------------------------------------------------|
| `Years_of_Experience`  | Right-skewed distribution (most candidates are early/mid career) |
| `Education_Level`      | High School / Diploma / Bachelor's / Master's / PhD |
| `Number_of_Skills`     | Count, loosely correlated with experience & education |
| `Certifications`       | Count, loosely correlated with experience & education |
| `Projects`             | Count, loosely correlated with experience & education |
| `Resume_Score`         | **Target.** A weighted combination of the above + random noise, rescaled to 0–100 |

The features aren't independent — more experienced, more educated candidates tend to have accumulated more skills/certifications/projects, just like in the real world. That correlation, plus the added noise, is what makes this a genuine (if small) regression problem rather than a trivial lookup.

## Model architecture

A small feed-forward ANN, exactly as specified for this project:

```
Input(5 features)
   ↓
Dense(64, ReLU)
   ↓
Dropout(0.2)
   ↓
Dense(32, ReLU)
   ↓
Dense(1, Linear)
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Regularization:** Dropout(0.2) to reduce overfitting on a fairly small dataset
- **Early stopping:** training stops once validation loss stops improving (patience = 12), and the best-performing weights are restored

Features are standardized with scikit-learn's `StandardScaler` before training — this matters a lot for neural nets, since `Years_of_Experience` and `Certifications` live on very different scales.

### Results from the included training run

| Metric | Value |
|--------|-------|
| MAE    | 5.99  |
| RMSE   | 7.51  |
| R²     | 0.911 |

(Your exact numbers will vary slightly if you regenerate the dataset/retrain, since the synthetic data uses random noise.)

## Score categories

Thresholds are calibrated against the actual distribution of predicted scores (not arbitrary round numbers), so they stay meaningful — roughly the top 10% of candidates land in "Excellent," the next 15% in "Good," and so on:

| Score range | Category           |
|-------------|---------------------|
| 80 – 100    | 🟢 Excellent         |
| 60 – 79     | 🔵 Good              |
| 35 – 59     | 🟡 Average           |
| 0 – 34      | 🔴 Needs Improvement |

## Getting started

### Prerequisites

- Python 3.10+ (developed and tested on 3.12)
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Resume-Screening-System.git
cd Resume-Screening-System

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the project

```bash
# 1. Train the model (generates the dataset on first run, then trains and saves model.h5)
python train_model.py

# 2. Launch the web app
streamlit run app.py
```

Streamlit will open the app in your browser (usually `http://localhost:8501`). If you skip step 1, `app.py` will tell you to run `train_model.py` first.

### Using the app

1. Go to the **🔍 Screen a Candidate** tab.
2. Fill in the candidate's name, years of experience, education level, and comma-separated skills / certifications / projects (optionally attach their resume PDF).
3. Click **Predict Resume Score**.
4. The predicted score, category, and (if a PDF was uploaded) an extracted text preview appear on the right, and the result is saved automatically.
5. Switch to **📊 Screening History** to see every past screening, and download it as CSV if you like.

## Pushing to your own GitHub

```bash
git init
git add .
git commit -m "Initial commit: Intelligent Resume Screening System"
git branch -M main
git remote add origin https://github.com/<your-username>/Resume-Screening-System.git
git push -u origin main
```

## Future improvements

Ideas for extending this beyond the current scope:

- Swap the synthetic dataset for a real, labeled resume dataset if one becomes available.
- Add lightweight NLP (e.g. keyword/skill extraction from the uploaded PDF) as a *separate, clearly-labeled* feature set, without changing the core ANN.
- Hyperparameter tuning (layer sizes, dropout rate, learning rate) with Keras Tuner.
- Model explainability (e.g. SHAP values) to show which factors drove a given score.
- Wrap the model in a small FastAPI service so it can be called from other tools, not just the Streamlit UI.
- User authentication, so screening history is scoped per recruiter.
- Deploy to Streamlit Community Cloud for a live demo link on your resume.

## License

MIT — see [LICENSE](LICENSE). Replace `[Your Name]` in the license file with your own name.

## Author

Built by **[Your Name]** — MCA project / portfolio piece.
Feel free to fork, adapt, and extend.
