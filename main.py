import os
import sys
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn
import requests
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load configuration from environment variables
S3_BUCKET = os.getenv('S3_BUCKET')
S3_KEY = os.getenv('S3_KEY', 'models/diabetes_model.joblib')
DATASET_URI = os.getenv('DATASET_URI')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'diabetes-classification')
MODEL_NAME = os.getenv('MODEL_NAME', 'diabetes-rf-model')

print("\n=== MLflow Training with Diagnostics ===\n")

# DIAGNOSTIC SECTION 1: Validate environment and connection
print("🔍 DIAGNOSTIC: Checking environment and MLflow connectivity")

# Validate required parameters
if not all([S3_BUCKET, DATASET_URI, MLFLOW_TRACKING_URI]):
    missing = [v for v in ['S3_BUCKET', 'DATASET_URI', 'MLFLOW_TRACKING_URI']
               if not os.getenv(v)]
    print(f"❌ Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

# Test MLflow server connection
print(f"Testing connection to MLflow server at {MLFLOW_TRACKING_URI}...")
try:
    response = requests.get(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list")
    if response.status_code == 200:
        print(f"✅ Successfully connected to MLflow server")
        experiments = response.json().get("experiments", [])
        print(f"   Server has {len(experiments)} experiments")
    else:
        print(f"❌ Connection failed with status code: {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Failed to connect to MLflow server: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Step 1: Configure MLflow
print(f"\nConfiguring MLflow with tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# DIAGNOSTIC SECTION 2: Check experiment creation
print("\n🔍 DIAGNOSTIC: Checking experiment setup")
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created new experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
    else:
        print(f"✅ Using existing experiment '{EXPERIMENT_NAME}' with ID: {experiment.experiment_id}")
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"❌ Failed to create/access experiment: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Load and prepare data
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

# Step 3: Train model and track with MLflow
print(f"\nStarting MLflow run under experiment: {EXPERIMENT_NAME}")
try:
    with mlflow.start_run(run_name=f"{MODEL_NAME}-training") as run:
        run_id = run.info.run_id
        print(f"✅ Started MLflow run with ID: {run_id}")

        # Log training parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        print("✅ Logged parameters")

        # Train the model
        print("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Model training complete")

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        print(f"Logging metrics - Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("✅ Logged metrics")

        # DIAGNOSTIC SECTION 3: Check model registry access
        print("\n🔍 DIAGNOSTIC: Testing model registry access")
        try:
            # List current models in registry
            from mlflow.tracking.client import MlflowClient
            client = MlflowClient()
            registered_models = client.list_registered_models()
            print(f"✅ Successfully accessed model registry, found {len(registered_models)} registered models")

            # List existing models (if any)
            existing_models = [model.name for model in registered_models]
            if MODEL_NAME in existing_models:
                print(f"   Note: Model '{MODEL_NAME}' already exists in registry")
        except Exception as e:
            print(f"⚠️ Warning: Could not access model registry: {str(e)}")
            print("   This might indicate permission issues with the model registry")
            # Continue execution since this is a diagnostic check

        # Create model signature and input example
        print("\nCreating model signature...")
        from mlflow.models.signature import infer_signature

        # Get predictions for signature
        signature_predictions = model.predict(X_train.iloc[:5])

        # Create signature from input and output
        signature = infer_signature(
            X_train.iloc[:5],  # Input sample
            signature_predictions  # Output sample
        )
        print("✅ Created model signature")

        # Log the model to MLflow with signature and example
        print(f"Logging model '{MODEL_NAME}' to MLflow with signature and input example...")
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            print(f"✅ Successfully logged model '{MODEL_NAME}' to MLflow")

            # Verify model was registered
            try:
                registered_model = client.get_registered_model(MODEL_NAME)
                print(f"✅ Verified model '{MODEL_NAME}' is in registry with {len(registered_model.latest_versions)} versions")
            except Exception as e:
                print(f"⚠️ Warning: Could not verify model registration: {str(e)}")
        except Exception as e:
            print(f"❌ Failed to log model: {str(e)}")
            traceback.print_exc()
except Exception as e:
    print(f"❌ Error during MLflow run: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Also save model to S3 directly
print("\nSaving model locally...")
local_path = '/tmp/model.joblib'
joblib.dump(model, local_path)

print(f"Uploading model to S3: s3://{S3_BUCKET}/{S3_KEY}")
try:
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, S3_BUCKET, S3_KEY)
    print(f"✅ Model also uploaded directly to S3: s3://{S3_BUCKET}/{S3_KEY}")
except Exception as e:
    print(f"⚠️ Direct S3 upload failed: {str(e)}")
    print("  (This is optional since the model is already stored in MLflow)")
finally:
    if os.path.exists(local_path):
        os.remove(local_path)

# Print summary and access instructions
print("\n=== Summary ===")
print(f"✅ Model trained with accuracy: {accuracy:.4f}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id if experiment else "unknown"

print("\n=== MLflow Access Information ===")
print(f"MLflow Server: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
print(f"Run ID: {run_id}")
print(f"Model Name: {MODEL_NAME}")

print("\n=== How to Access in MLflow UI ===")
print("1. To view experiment and run:")
print(f"   {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}")
print("2. To view model in registry:")
print(f"   {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")

print("\n=== Troubleshooting Tips ===")
print("- If you can't see the model in the UI, check:")
print("  • Are you looking in the Models tab of the left sidebar?")
print("  • Do you have the correct permissions to the MLflow server?")
print("  • Is the MLflow server configured to allow model registration?")
print("  • Are there any firewall issues preventing artifact uploads?")