"""
==================
Melatih model Machine Learning menggunakan MLflow autolog (Basic)
Dataset: ASAP2_train

Usage:
    python modelling.py

pastikan sudah menjalankan perintah berikut untuk memulai MLflow server:
    mlflow ui --host
"""

import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
TRAIN_PATH = "asap2_preprocessing/train_preprocessed.csv"
TEST_PATH = "asap2_preprocessing/test_preprocessed.csv"
EXPERIMENT_NAME = "ASAP2_Essay_Score_Classification"
RANDOM_STATE = 42

# ---------------------------------------------------
# Setup MLflow
# ---------------------------------------------------
mlflow.set_tracking_uri(
    "http://localhost:5000"
)  # Pastikan MLflow server berjalan di localhost:5000
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
def load_data(train_path, test_path):
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["processed_text"].fillna("")
    y_train = train_df["score_label"].astype(str)
    X_test = test_df["processed_text"].fillna("")
    y_test = test_df["score_label"].astype(str)

    print(f"Train data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
MODELS = {
    "GradientBoostingClassifier": Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "clf",
                GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            ),
        ]
    ),
}


# ---------------------------------------------------
# Train with MLflow autolog
# ---------------------------------------------------
def train_mode(name, pipeline, X_train, X_test, y_train, y_test):
    print(f"Training model: {name}")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        class_report = classification_report(y_test, y_pred)
        precision_score_ = precision_score(y_test, y_pred, average="weighted")
        recall_score_ = recall_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        print(f"Precision: {precision_score_:.4f} | Recall: {recall_score_:.4f}")
        print("Classification Report:")
        print(class_report)

    print(f"Finished training {name}\n")


# ---------------------------------------------------
# Main function
# ---------------------------------------------------
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)

    for name, pipeline in MODELS.items():
        train_mode(name, pipeline, X_train, X_test, y_train, y_test)

    print("All models trained and logged to MLflow!")
