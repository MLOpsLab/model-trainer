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

# Load configuration from environment variables
S3_BUCKET = os.getenv('S3_BUCKET')
S3_KEY = os.getenv('S3_KEY', 'models/diabetes_model.joblib')
DATASET_URI = os.getenv('DATASET_URI')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'diabetes-experiment')
MODEL_NAME = os.getenv('MODEL_NAME', 'diabetes-rf-model')

print("\n=== MLflow Training (v2.22.0) ===\n")
print(f"MLflow version: {mlflow.__version__}")

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
print(f"- Experiment: {EXPERIMENT_NAME}")
print(f"- Model name: {MODEL_NAME}")

# Step 1: Configure MLflow
print(f"\nConfiguring MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Step 2: Create a brand new experiment every time (avoid using default or existing ones)
print(f"Creating a new experiment...")
timestamp = int(time.time())
new_experiment_name = f"{EXPERIMENT_NAME}-{timestamp}"
try:
    experiment_id = mlflow.create_experiment(new_experiment_name)
    print(f"✅ Created new experiment '{new_experiment_name}' with ID: {experiment_id}")
    EXPERIMENT_NAME = new_experiment_name
except Exception as e:
    print(f"❌ Failed to create experiment: {e}")
    # Try another name as a fallback
    fallback_name = f"experiment-{timestamp}"
    try:
        experiment_id = mlflow.create_experiment(fallback_name)
        print(f"✅ Created fallback experiment '{fallback_name}' with ID: {experiment_id}")
        EXPERIMENT_NAME = fallback_name
    except Exception as e2:
        print(f"❌ Failed to create fallback experiment: {e2}")
        print("⚠️ Will proceed without experiment tracking")
        experiment_id = None

# Explicitly set the experiment using experiment_id, not name
if experiment_id is not None:
    try:
        mlflow.set_experiment(experiment_id=experiment_id)
        print(f"✅ Set active experiment to ID: {experiment_id}")
    except Exception as e:
        print(f"❌ Failed to set experiment: {e}")
        print("⚠️ Will proceed without experiment tracking")

# Step 3: Load and prepare data
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

# Step 4: Train model and track with MLflow
print(f"\nTraining model...")
# Train the model (do this outside MLflow in case MLflow fails)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Create model signature
from mlflow.models.signature import infer_signature
signature = infer_signature(
    X_train.iloc[:5],  # Input sample
    model.predict(X_train.iloc[:5])  # Output sample
)

# Only log to MLflow if we have a valid experiment
run_id = None
if experiment_id is not None:
    print(f"\nStarting MLflow run...")
    try:
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{MODEL_NAME}-training") as run:
            run_id = run.info.run_id
            print(f"✅ Started run with ID: {run_id}")

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

            # Log the model to MLflow
            print(f"Logging model '{MODEL_NAME}' to MLflow...")
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                    signature=signature,
                    input_example=X_train.iloc[:5]
                )
                print(f"✅ Model '{MODEL_NAME}' successfully logged to MLflow")
            except Exception as e:
                print(f"⚠️ Warning: Issue with model logging: {e}")
                print("⚠️ Continuing with S3 upload only")
    except Exception as e:
        print(f"❌ Error during MLflow run: {str(e)}")
        traceback.print_exc()
        print("⚠️ Continuing with S3 upload only")
else:
    print("⚠️ Skipping MLflow logging due to experiment creation failure")

# Step 5: Save model to S3 directly (always do this as a backup)
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

# Print summary
print("\n=== Summary ===")
print(f"✅ Model trained with accuracy: {accuracy:.4f}")
print(f"✅ Model saved to S3: s3://{S3_BUCKET}/{S3_KEY}")

if experiment_id is not None and run_id is not None:
    print("\n=== MLflow Access Information ===")
    print(f"MLflow Server: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")
    print(f"Model Name: {MODEL_NAME}")

    print("\n=== How to Access in MLflow UI ===")
    print("1. To view experiment and run:")
    print(f"   {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}")
    print("2. To view model in registry:")
    print(f"   {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")

    print("\n=== MLflow Model URI for Serving ===")
    print(f"Run URI: runs:/{run_id}/model")
    print(f"Registry URI: models:/{MODEL_NAME}/latest")

print("\n=== Training Complete ===")