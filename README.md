# Resume Screening System

A small ANN project I built to predict a "resume score" (0-100) from a candidate's experience, education, skills, certifications, and project count. Built with TensorFlow/Keras and a Streamlit front end.

## Why I built this

I wanted a project that actually goes through a full deep learning workflow - data, preprocessing, a real trained model, evaluation, and a working UI - instead of just a notebook that ends at a metric. Resume scoring felt like a simple enough problem to keep the focus on the ANN itself rather than getting lost in NLP.

## What it does

- Enter a candidate's years of experience, education level, and comma-separated skills/certifications/projects
- Optionally attach their resume PDF (stored for reference, previewed as text - it doesn't feed the model)
- Get a predicted Resume Score (0-100) and a category: Excellent / Good / Average / Needs Improvement
- Every screening gets saved to SQLite, browsable in a history tab, exportable as CSV
- Sidebar shows the model's MAE / RMSE / R² from the last training run

## Why it's a simple, structured-data model and not NLP

The "resume" here is five numbers, not raw text. That keeps the actual deep learning parts (architecture, loss function, optimizer, regularization, evaluation) front and center, which was the point of the exercise. Parsing actual resume text is a different, bigger problem I didn't want to solve at the same time.

## Dataset

I'm using [this dataset from Kaggle](https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025), which has experience, education, skills, certifications, project count, and an AI score for each candidate. I mapped it onto the same five features my model uses - counting the comma-separated skills/certifications lists and using the project count directly.

(Like most "resume score" data, this is synthetic - there's no real ground-truth dataset for this that I could find. But it's an independently published one rather than something I generated myself, so at least the noise/correlations in it aren't ones I hand-picked.)

## Model

Input(5 features)
↓
Dense(64, ReLU)
↓
Dropout(0.2)
↓
Dense(32, ReLU)
↓
Dense(1, Linear)


- Loss: MSE, Optimizer: Adam
- Dropout(0.2) since the dataset isn't huge
- Early stopping (patience=12), restoring the best weights
- Features are scaled with StandardScaler before training - experience and certifications live on very different numeric ranges, so this matters for a neural net

## Results

| Metric | Value |
|---|---|
| MAE | 5.99 |
| RMSE | 7.51 |
| R² | 0.911 |

## Project structure

app.py # Streamlit app - the UI
train_model.py # loads the dataset, preprocesses, trains and saves the model
model.h5 # trained model
scaler.pkl # fitted scaler, needed to preprocess new inputs the same way
metrics.json # last training run's metrics
utils/
preprocess.py # encoding/scaling logic shared by app.py and train_model.py
database.py # SQLite helpers
dataset/
resumes.csv # training data


## Running it locally

git clone https://github.com/akshaytp8/Resume-Screening-System.git
cd Resume-Screening-System
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train_model.py # trains the model, only needed once (or whenever you retrain)
streamlit run app.py


## What I'd improve if I kept working on this

- Use real resume text with some lightweight NLP for skill extraction instead of manually typing skills in
- Try hyperparameter tuning instead of the architecture I picked by hand
- Add model explainability (SHAP) to show which factor actually drove a given score
- Wrap the model in a small API so it isn't tied to the Streamlit UI

## Tech used

Python, TensorFlow/Keras, scikit-learn, Pandas, Streamlit, SQLite

---
Built by Akshay T P.
