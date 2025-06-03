import os
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn
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

# Validate required parameters
if not all([S3_BUCKET, DATASET_URI, MLFLOW_TRACKING_URI]):
    missing = [v for v in ['S3_BUCKET', 'DATASET_URI', 'MLFLOW_TRACKING_URI']
               if not os.getenv(v)]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Step 1: Configure MLflow
print(f"Configuring MLflow with tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Step 2: Load and prepare data
print(f"Loading dataset from: {DATASET_URI}")
df = pd.read_csv(DATASET_URI)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model and track with MLflow
print(f"Starting MLflow run under experiment: {EXPERIMENT_NAME}")
with mlflow.start_run(run_name=f"{MODEL_NAME}-training"):
    # Log training parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # Train the model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

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

    # Create model signature and input example
    print("Creating model signature...")
    from mlflow.models.signature import infer_signature

    # Get predictions for signature
    signature_predictions = model.predict(X_train.iloc[:5])

    # Create signature from input and output
    signature = infer_signature(
        X_train.iloc[:5],  # Input sample
        signature_predictions  # Output sample
    )

    # Create input example
    input_example = X_train.iloc[:5].to_dict(orient="split")

    # Log the model to MLflow with signature and example
    print("Logging model to MLflow with signature and input example...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature,  # Add signature
        input_example=X_train.iloc[:5]  # Add input example
    )

    # Get the run ID
    run_id = mlflow.active_run().info.run_id
    print(f"MLflow run ID: {run_id}")
# Step 4: Also save model to S3 directly (optional, since MLflow already saves artifacts)
print("Saving model locally...")
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

# Print summary
print("\n=== Summary ===")
print(f"✅ Model trained with accuracy: {accuracy:.4f}")
print(f"✅ Run tracked in MLflow: {MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")
print(f"✅ Model registered in MLflow Model Registry as: {MODEL_NAME}")