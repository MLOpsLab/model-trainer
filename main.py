import pandas as pd
from fastapi import FastAPI
import uvicorn
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Environment Variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
ALIAS = os.getenv('ALIAS')
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_URI = os.getenv('DATASET_URI')
ARTIFACT_URI = os.getenv('ARTIFACT_URI')

# Verify the environment variables
if not all([MLFLOW_TRACKING_URI, ALIAS, MODEL_NAME, DATASET_URI, ARTIFACT_URI]):
    raise ValueError("One or more required environment variables are missing.")

# Configure MLflow to use S3
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Default")  # Ensure experiment exists

def train_and_log_model():
    # 1. Download remote dataset (Pima Indian Diabetes)
    try:
        df = pd.read_csv(DATASET_URI)
    except Exception as e:
        raise ValueError(f"Error loading dataset from {DATASET_URI}: {str(e)}")

    # 2. Prepare data
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Evaluate the model
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # 5. MLflow Logging
    with mlflow.start_run(run_name=MODEL_NAME):
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Log model to S3
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path=f"models:/{MODEL_NAME}/dev",  # S3 path for the model
            registered_model_name=MODEL_NAME
        )

    # 6. Get and alias the model version
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Find the latest model version
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if model_versions:
        latest_version = max(int(mv.version) for mv in model_versions)

        # Set alias for the latest version
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS,
            version=latest_version
        )

        print(f"Model {MODEL_NAME} logged with version {latest_version}")

    return rf, acc

# FastAPI app setup
app = FastAPI()

@app.on_event("startup")
def startup_event():
    try:
        train_and_log_model()
    except Exception as e:
        print(f"Model training failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)