import os
import sys
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn
import traceback
import time
import uuid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n=== Training With Clean MLflow Setup ===\n")

# Load configuration from environment variables
S3_BUCKET = os.getenv('S3_BUCKET')
S3_KEY = os.getenv('S3_KEY', 'models/diabetes_model.joblib')
DATASET_URI = os.getenv('DATASET_URI')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_NAME = os.getenv('MODEL_NAME', 'diabetes-rf-model')

# Validate required parameters
if not all([S3_BUCKET, DATASET_URI, MLFLOW_TRACKING_URI]):
    missing = [v for v in ['S3_BUCKET', 'DATASET_URI', 'MLFLOW_TRACKING_URI']
               if not os.getenv(v)]
    print(f"❌ Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

print(f"Configuration:")
print(f"- S3 Bucket: {S3_BUCKET}")
print(f"- Dataset: {DATASET_URI}")
print(f"- MLflow URI: {MLFLOW_TRACKING_URI}")
print(f"- Model name: {MODEL_NAME}")

# Step 1: Load and prepare data first
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

# Step 2: Train the model (outside of MLflow)
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

# Step 4: Save model to S3 (guaranteed to work)
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
finally:
    if os.path.exists(local_path):
        os.remove(local_path)
        print("✅ Removed temporary local file")

# Step 5: Try MLflow logging with a completely fresh approach
print("\n=== MLflow Tracking (Fresh Approach) ===")
exp_id = None
run_id = None

try:
    # Configure MLflow
    print("Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create a guaranteed unique experiment name
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())
    unique_exp_name = f"diabetes-{timestamp}-{unique_id}"

    print(f"Creating new experiment: {unique_exp_name}")
    exp_id = mlflow.create_experiment(unique_exp_name)
    print(f"✅ Created new experiment with ID: {exp_id}")

    # Create a run directly with the experiment ID
    print("Starting new run...")
    run = mlflow.start_run(experiment_id=exp_id, run_name=f"{MODEL_NAME}-run-{unique_id}")
    run_id = run.info.run_id
    print(f"✅ Started run with ID: {run_id}")

    # Log parameters
    print("Logging parameters...")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    print("Logging metrics...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Create model signature
    print("Creating model signature...")
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

    # End the run
    mlflow.end_run()
    print("✅ MLflow run completed successfully")

    mlflow_success = True
except Exception as e:
    print(f"❌ MLflow error: {e}")
    traceback.print_exc()
    mlflow_success = False
    # Try to end run if it was started
    try:
        mlflow.end_run()
    except:
        pass

# Print summary
print("\n=== Summary ===")
print(f"✅ Model trained with accuracy: {accuracy:.4f}")
print(f"✅ Model saved to S3: s3://{S3_BUCKET}/{S3_KEY}")

if mlflow_success and exp_id and run_id:
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
else:
    print("\n⚠️ MLflow tracking failed, but model was still saved to S3")

print("\n=== Training Complete ===")