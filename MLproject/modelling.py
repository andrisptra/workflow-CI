"""
modelling_tuning.py
====================
Melatih model ML dengan:
  - Hyperparameter tuning (GridSearchCV)
  - Manual logging MLflow (bukan autolog)
  - DagsHub remote tracking
  - Artefak tambahan: confusion matrix plot + classification report JSON

Dataset: ASAP2 Essay Score Classification

Setup DagsHub:
    1. Buat akun di https://dagshub.com
    2. Buat repo baru di DagsHub
    3. Isi DAGSHUB_USERNAME dan DAGSHUB_REPO_NAME di bawah
    4. Set environment variable:
       export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
       export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

Usage:
    python modelling_tuning.py
"""

import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend for plotting
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "apq000d6y0649")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "ASAP2_Essay_Score")

DAGSHUB_TRACKING_URI = (
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
)

TRAIN_PATH = "asap2_preprocessing/train_preprocessed.csv"
TEST_PATH = "asap2_preprocessing/test_preprocessed.csv"
EXPERIMENT_NAME = "ASAP2_Essay_Score_Tuning"
RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------
# Setup MLflow -> DagsHub Remote Tracking
# ---------------------------------------------------
def setup_mlflow():
    mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI set to: {DAGSHUB_TRACKING_URI}")


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
def load_data(train_path, test_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train["processed_text"].fillna("")
    y_train = train["score_label"].astype(str)
    X_test = test["processed_text"].fillna("")
    y_test = test["score_label"].astype(str)

    print(f"Train data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------
# Artifact 1: Confusion Matrix Plot
# ---------------------------------------------------
def save_confussion_matrix(y_test, y_pred, model_name: str) -> str:
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=16)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


# ---------------------------------------------------
# Artifact 2: Classification Report JSON
# ---------------------------------------------------
def save_classification_report(y_test, y_pred, model_name: str) -> str:
    report = classification_report(y_test, y_pred, output_dict=True)
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_classification_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    return path


# ---------------------------------------------------
# Artifact 3: Feature Importantce plot (for tree models)
# ---------------------------------------------------


def save_feature_importance(pipeline, model_name: str, top_n: int = 20) -> str:
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            return None

        indices = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), importances[indices], align="center", color="skyblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=12)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()

        path = os.path.join(ARTIFACTS_DIR, f"feature_importance_{model_name}.png")
        plt.savefig(path, dpi=100)
        plt.close()
        return path
    except Exception as e:
        print(f" [warning] Could not save feature importance: {e}")
        return None


# ---------------------------------------------------
# Models + Hyperparameter Grids
# ---------------------------------------------------
MODEL_CONFIGS = {
    "GradientBoostingClassifier": {
        "pipeline": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=100, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "param_grid": {
            "tfidf__max_features": [3000, 5000],
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
        },
    },
}


# ---------------------------------------------------
# Train + Tune + Log with MLflow
# ---------------------------------------------------


def train_and_tune(name, config, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 50}")
    print(f"Training and tuning model: {name}")
    print(f"{'=' * 50}\n")

    pipeline = config["pipeline"]
    param_grid = config["param_grid"]

    # Hyperparameter tuning with GridSearchCV
    print(" Starting GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring="f1_weighted"
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    print(f" Best CV F1     : {best_cv_f1:.4f}")
    print(f" Best params    : {best_params}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision_score_ = precision_score(y_test, y_pred, average="weighted")
    recall_score_ = recall_score(y_test, y_pred, average="weighted")

    print(f" Test Accuracy  : {acc:.4f}")
    print(f" Test F1 Score  : {f1:.4f}")
    print(f" Test Precision : {precision_score_:.4f}")
    print(f" Test Recall    : {recall_score_:.4f}")

    # Save artifacts
    cm_path = save_confussion_matrix(y_test, y_pred, name)
    report_path = save_classification_report(y_test, y_pred, name)
    fi_path = save_feature_importance(best_model, name)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{name}_Tuned"):
        # log Hyperparameter
        mlflow.log_params({"model": name, "cv_folds": 3, "scoring": "f1_weighted"})
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # log Metrics
        mlflow.log_metrics(
            {
                "test_accuracy": acc,
                "test_f1": f1,
                "test_precision": precision_score_,
                "test_recall": recall_score_,
                "best_cv_f1": best_cv_f1,
            }
        )

        # Log per-class F1
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                safe_label = label.replace(" ", "_")
                mlflow.log_metric(f"f1_{safe_label}", metrics.get("f1-score", 0))
                mlflow.log_metric(
                    f"precision_{safe_label}", metrics.get("precision", 0)
                )
                mlflow.log_metric(f"recall_{safe_label}", metrics.get("recall", 0))

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Log extra artifacts
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="reports")
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # Log tags
        mlflow.set_tag("dataset", "ASAP2_Essay_Score")
        mlflow.set_tag("task", "multiclass_classification")
        mlflow.set_tag("tuned", "True")
        mlflow.set_tag("developer", DAGSHUB_USERNAME)

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow Run ID: {run_id}")

    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "precision": precision_score_,
        "recall": recall_score_,
    }


# ---------------------------------------------------
# Main Execution
# ---------------------------------------------------
if __name__ == "__main__":
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)

    results = []
    for name, config in MODEL_CONFIGS.items():
        result = train_and_tune(name, config, X_train, y_train, X_test, y_test)
        results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("  SUMMARY — Model Comparison")
    print("=" * 55)
    df_results = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(df_results.to_string(index=False))

    best = df_results.iloc[0]
    print(
        f"\n🏆 Best Model: {best['model']} | F1: {best['f1']:.4f} | Accuracy: {best['accuracy']:.4f}"
    )
    print(f"\n✅ All runs logged to DagsHub MLflow: {DAGSHUB_TRACKING_URI}")
