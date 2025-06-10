import os
import sys
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn
import traceback
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n=== Training Without Default MLflow Experiment ===\n")

# Load configuration from environment variables
S3_BUCKET = os.getenv('S3_BUCKET')
S3_KEY = os.getenv('S3_KEY', 'models/diabetes_model.joblib')
DATASET_URI = os.getenv('DATASET_URI')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_NAME = os.getenv('MODEL_NAME', 'diabetes-rf-model')

# Validate required parameters
if not all([S3_BUCKET, DATASET_URI]):
    missing = [v for v in ['S3_BUCKET', 'DATASET_URI'] if not os.getenv(v)]
    print(f"❌ Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

print(f"Configuration:")
print(f"- S3 Bucket: {S3_BUCKET}")
print(f"- Dataset: {DATASET_URI}")
print(f"- MLflow URI: {MLFLOW_TRACKING_URI}")
print(f"- Model name: {MODEL_NAME}")

# Step 1: Load and prepare data
print(f"\nLoading dataset from: {DATASET_URI}")
try:
    df = pd.read_csv(DATASET_URI)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Successfully loaded dataset with {len(X_train)} training samples, {len(X_test)} test samples")
except Exception as e:
    print(f"❌ Failed to load or process dataset: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Train the model
print(f"\nTraining RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model metrics:")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")

# Step 4: Save model to S3
print("\nSaving model to S3...")
local_path = '/tmp/model.joblib'
joblib.dump(model, local_path)

print(f"Uploading model to S3: s3://{S3_BUCKET}/{S3_KEY}")
try:
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, S3_BUCKET, S3_KEY)
    print(f"✅ Model successfully uploaded to S3: s3://{S3_BUCKET}/{S3_KEY}")
except Exception as e:
    print(f"❌ S3 upload failed: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if os.path.exists(local_path):
        os.remove(local_path)
        print("✅ Removed temporary local file")

# Step 5: Try MLflow logging separately after successful S3 upload
if MLFLOW_TRACKING_URI:
    print("\nAttempting MLflow logging as a separate step...")

    try:
        # Set up MLflow with completely new experiment
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Create a unique experiment name with timestamp
        timestamp = int(time.time())
        unique_exp_name = f"diabetes-exp-{timestamp}"

        try:
            # Create a new experiment explicitly
            exp_id = mlflow.create_experiment(unique_exp_name)
            print(f"✅ Created new MLflow experiment: {unique_exp_name} (ID: {exp_id})")

            # Start a run directly with the new experiment ID
            with mlflow.start_run(experiment_id=exp_id, run_name=f"{MODEL_NAME}-run") as run:
                run_id = run.info.run_id
                print(f"✅ Started MLflow run with ID: {run_id}")

                # Log parameters
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("test_size", 0.2)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Create model signature
                from mlflow.models.signature import infer_signature
                signature = infer_signature(
                    X_train.iloc[:5],
                    model.predict(X_train.iloc[:5])
                )

                # Log the model to MLflow
                print(f"Logging model '{MODEL_NAME}' to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                    signature=signature,
                    input_example=X_train.iloc[:5]
                )
                print(f"✅ Model '{MODEL_NAME}' successfully logged to MLflow")

                # Print MLflow access information
                print("\n=== MLflow Access Information ===")
                print(f"MLflow Server: {MLFLOW_TRACKING_URI}")
                print(f"Experiment: {unique_exp_name}")
                print(f"Experiment ID: {exp_id}")
                print(f"Run ID: {run_id}")
                print(f"Model Name: {MODEL_NAME}")
                print(f"\nUI URLs:")
                print(f"- Experiment: {MLFLOW_TRACKING_URI}/#/experiments/{exp_id}")
                print(f"- Run: {MLFLOW_TRACKING_URI}/#/experiments/{exp_id}/runs/{run_id}")
                print(f"- Model: {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")

                print(f"\nModel URIs for serving:")
                print(f"- Run URI: runs:/{run_id}/model")
                print(f"- Registry URI: models:/{MODEL_NAME}/latest")
        except Exception as e:
            print(f"❌ MLflow experiment/run error: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ MLflow general error: {e}")
        traceback.print_exc()

    print("\nNote: MLflow logging was attempted separately and doesn't affect the model in S3")
else:
    print("\nSkipping MLflow logging (MLFLOW_TRACKING_URI not set)")

# Print summary
print("\n=== Summary ===")
print(f"✅ Model trained with accuracy: {accuracy:.4f}")
print(f"✅ Model saved to S3: s3://{S3_BUCKET}/{S3_KEY}")
if MLFLOW_TRACKING_URI:
    print(f"ℹ️ MLflow logging was attempted separately")

print("\n=== Training Complete ===")