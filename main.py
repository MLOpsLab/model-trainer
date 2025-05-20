import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
ALIAS = os.getenv('ALIAS')
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_URI = os.getenv('DATASET_URI')
ARTIFACT_URI = os.getenv('ARTIFACT_URI')

app = FastAPI()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Set the artifact location (S3 bucket where artifacts will be stored)
artifact_location = ARTIFACT_URI

# Create a new experiment with the specified artifact location (S3 bucket)
experiment_id = mlflow.create_experiment("my_experiment", artifact_location=artifact_location)

# Set the experiment to use
mlflow.set_experiment(experiment_id)


# 1. Download remote dataset (Pima Indian Diabetes)
# Directly read from the remote CSV (no storing to disk)
df = pd.read_csv(DATASET_URI)

# The following lines are still required:
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train model and log with MLflow if not already
runs = mlflow.search_runs(experiment_names=["Default"], filter_string=f'tags.mlflow.runName="{MODEL_NAME}"')
if runs.shape[0] == 0:
    with mlflow.start_run(run_name=MODEL_NAME):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model", registered_model_name=MODEL_NAME)
    # Get latest version number for the model
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(int(mv.version) for mv in model_versions)
    # Assign the alias to the latest version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=latest_version
    )
else:
    print("Model already tracked in MLflow.")