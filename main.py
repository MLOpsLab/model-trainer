import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
ALIAS = os.getenv('ALIAS')
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_URI = os.getenv('DATASET_URI')
ARTIFACT_URI = os.getenv('ARTIFACT_URI')  # s3://bucket/path/

print("Environment Variables:")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("ARTIFACT_URI:", ARTIFACT_URI)
print("MODEL_NAME:", MODEL_NAME)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create a NEW experiment with S3 artifact location
experiment_name = f"S3_Experiment_{MODEL_NAME}"

try:
    # Delete experiment if it exists (for testing)
    try:
        existing = mlflow.get_experiment_by_name(experiment_name)
        if existing and existing.lifecycle_stage != "deleted":
            client = MlflowClient()
            client.delete_experiment(existing.experiment_id)
            print(f"Deleted existing experiment")
    except:
        pass

    # Create fresh experiment with S3 artifact store
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=ARTIFACT_URI
    )
    print(f"✅ Created experiment with S3 artifacts: {ARTIFACT_URI}")

except Exception as e:
    print(f"Using existing experiment: {e}")

mlflow.set_experiment(experiment_name)

def train_and_log_model():
    df = pd.read_csv(DATASET_URI)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    signature = infer_signature(X_train, preds)
    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=MODEL_NAME):
        # Check where artifacts will go
        current_run = mlflow.active_run()
        print(f"🏃 Run artifacts will go to: {current_run.info.artifact_uri}")

        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42
        })
        mlflow.log_metric("accuracy", acc)

        # Use simple artifact_path (not full S3 URL)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # Simple path, not S3 URL
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example
        )

        print(f"✅ Model logged to: {model_info.model_uri}")

        # Verify it's in S3
        if "s3://" in model_info.model_uri:
            print(f"🎉 SUCCESS! Model is in S3: {model_info.model_uri}")
        else:
            print(f"❌ Model not in S3: {model_info.model_uri}")

if __name__ == "__main__":
    train_and_log_model()