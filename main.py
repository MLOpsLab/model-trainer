import os
import pandas as pd
from fastapi import FastAPI
import uvicorn
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Get environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
ALIAS = os.getenv('ALIAS')
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_URI = os.getenv('DATASET_URI')
ARTIFACT_URI = os.getenv('ARTIFACT_URI')

# Debug prints to verify values
print("DEBUG ENV VARS:")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("ALIAS:", ALIAS)
print("MODEL_NAME:", MODEL_NAME)
print("DATASET_URI:", DATASET_URI)
print("ARTIFACT_URI:", ARTIFACT_URI)

# Verify all required env vars
if not all([MLFLOW_TRACKING_URI, ALIAS, MODEL_NAME, DATASET_URI, ARTIFACT_URI]):
    raise ValueError("One or more required environment variables are missing.")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Default")

def train_and_log_model():
    try:
        df = pd.read_csv(DATASET_URI)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {DATASET_URI}: {e}")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest_version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS,
            version=latest_version
        )
        print(f"Model registered with version {latest_version} and alias '{ALIAS}'")

    return model, acc

app = FastAPI()

@app.on_event("startup")
def on_startup():
    try:
        train_and_log_model()
    except Exception as e:
        print("Model training failed:", e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)