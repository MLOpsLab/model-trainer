import os
import pandas as pd
import mlflow
import mlflow.sklearn
import boto3
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tempfile
import shutil

# Load environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_URI = os.getenv('DATASET_URI')
S3_BUCKET = "mlops-lab-artifact"  # Your bucket name
S3_PREFIX = "mlflow-models"       # Folder in bucket

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Default")

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
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metric("accuracy", acc)

        # Log model normally first (to MLflow tracking)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example
        )

        print(f"📊 Model logged to MLflow: {model_info.model_uri}")

        # Now save a copy directly to S3
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model to temp directory
            local_path = os.path.join(temp_dir, "model")
            mlflow.sklearn.save_model(model, local_path, signature=signature, input_example=input_example)

            # Upload to S3
            s3_client = boto3.client('s3')
            s3_key_prefix = f"{S3_PREFIX}/{MODEL_NAME}/"

            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = s3_key_prefix + relative_path

                    s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
                    print(f"📤 Uploaded: s3://{S3_BUCKET}/{s3_key}")

            print(f"🎉 Model saved to S3: s3://{S3_BUCKET}/{s3_key_prefix}")

if __name__ == "__main__":
    train_and_log_model()