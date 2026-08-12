"""
train_model.py
---------------
Trains a simple feed-forward Artificial Neural Network (ANN) that predicts
a candidate's Resume Score (0-100) from five structured features:

    Years of Experience, Education Level, Number of Skills,
    Certifications, Projects

No public dataset is required: if dataset/resumes.csv doesn't exist yet,
a realistic SYNTHETIC dataset is generated automatically on first run.

Usage:
    python train_model.py

Outputs (all written to the project root):
    dataset/resumes.csv     - the training data (generated if missing)
    model.h5                - the trained Keras model
    scaler.pkl              - the fitted StandardScaler (needed at inference time)
    metrics.json             - MAE / RMSE / R2 on the held-out test set
    training_history.png     - training vs validation loss curve
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless - just save the plot, don't try to open a window
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.preprocess import EDUCATION_MAP

# ----------------------------------------------------------------------
# Reproducibility - same seed => same dataset & same trained weights
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "resumes.csv")
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.json"
HISTORY_PLOT_PATH = "training_history.png"


# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------
def generate_synthetic_dataset(n_samples=3000, random_seed=SEED):
    """
    Build a realistic synthetic resume dataset.

    Each feature uses a plausible real-world distribution, and
    Skills/Certifications/Projects are loosely correlated with Years of
    Experience and Education (more experienced/educated candidates tend
    to have done more) - just like in the real world. The Resume Score
    is a weighted combination of all five features plus random noise,
    rescaled into a realistic 0-100 band.
    """
    rng = np.random.default_rng(random_seed)

    # Years of experience: right-skewed - most candidates are early/mid career
    years_experience = np.round(rng.gamma(shape=2.0, scale=2.4, size=n_samples), 1)
    years_experience = np.clip(years_experience, 0, 25)

    # Education level: categorical, weighted towards Bachelor's / Master's
    education_levels = list(EDUCATION_MAP.keys())
    education_weights = [0.05, 0.15, 0.45, 0.28, 0.07]
    education = rng.choice(education_levels, size=n_samples, p=education_weights)
    education_encoded = np.array([EDUCATION_MAP[e] for e in education])

    # Skills / certifications / projects: Poisson counts whose average
    # rises a little with experience & education (realistic correlation,
    # not just independent random noise)
    num_skills = rng.poisson(lam=5 + education_encoded * 1.1 + years_experience * 0.25)
    num_skills = np.clip(num_skills, 1, 30)

    certifications = rng.poisson(lam=0.6 + years_experience * 0.15 + (education_encoded >= 4) * 0.8)
    certifications = np.clip(certifications, 0, 10)

    projects = rng.poisson(lam=2.0 + years_experience * 0.35 + education_encoded * 0.3)
    projects = np.clip(projects, 0, 20)

    # Weighted combination -> raw score, then rescaled to a 0-100 band
    raw_score = (
        years_experience * 2.0
        + education_encoded * 7.0
        + num_skills * 1.6
        + certifications * 3.2
        + projects * 2.0
        + rng.normal(0, 5, n_samples)  # noise, so it isn't perfectly predictable
    )

    # Rescale using the 3rd/97th percentile (not the raw min/max) so a
    # handful of extreme outliers (e.g. 25 years of experience) don't
    # compress everyone else into the bottom of the range. The middle
    # 94% of candidates get spread across the full 0-100 band, and only
    # the true extremes clip at 0 or 100 - which is how a real scoring
    # rubric behaves.
    lo, hi = np.percentile(raw_score, [3, 97])
    score = (raw_score - lo) / (hi - lo) * 100
    score = np.clip(np.round(score, 2), 0, 100)

    df = pd.DataFrame({
        "Years_of_Experience": years_experience,
        "Education_Level": education,
        "Number_of_Skills": num_skills,
        "Certifications": certifications,
        "Projects": projects,
        "Resume_Score": score,
    })

    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated synthetic dataset with {n_samples} rows -> {DATASET_PATH}")
    return df


def load_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Loading existing dataset from {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)
    print("No dataset found - generating a synthetic one...")
    return generate_synthetic_dataset()


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
def preprocess(df):
    """Encode Education_Level and split into X (features) / y (target)."""
    df = df.copy()
    df["Education_Level_Encoded"] = df["Education_Level"].map(EDUCATION_MAP)
    X = df[[
        "Years_of_Experience", "Education_Level_Encoded",
        "Number_of_Skills", "Certifications", "Projects",
    ]].values.astype(float)
    y = df["Resume_Score"].values.astype(float)
    return X, y


# ----------------------------------------------------------------------
# 3. Model - exactly the architecture from the project spec
# ----------------------------------------------------------------------
def build_model(input_dim):
    """
    Simple feed-forward ANN:
        Input -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, Linear)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ----------------------------------------------------------------------
# 4. Main training routine
# ----------------------------------------------------------------------
def main():
    df = load_dataset()
    print("\nDataset summary:")
    print(df.describe(include="all"))

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Feature scaling matters a lot for neural networks: without it, a
    # large-range feature like Years_of_Experience can dominate training
    # over a small-range one like Certifications.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # --------------------------------------------------------------
    # 5. Evaluation - the two metrics required by the spec (+ bonus R^2)
    # --------------------------------------------------------------
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== Evaluation on Test Set =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R^2  : {r2:.3f}")

    # --------------------------------------------------------------
    # 6. Save model, scaler, metrics, and a training curve plot
    # --------------------------------------------------------------
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Training curve saved to {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
