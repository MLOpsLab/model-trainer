import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature  # Add this import

# MLflow tracking server URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'diabetes-model')
DATASET_URI = os.getenv('DATASET_URI')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'diabetes-classification')

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or get experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Load and prepare data
df = pd.read_csv(DATASET_URI)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create model signature and input example
signature = infer_signature(X_train, y_pred)  # Create signature from input and output
input_example = X_train.iloc[:5].to_dict(orient="split")  # Create input example

# Log model with MLflow
with mlflow.start_run(run_name=f"{MODEL_NAME}-run"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature and input example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature,           # Add this
        input_example=X_train.iloc[:5] # Add this
    )

    # Get the run ID for reference
    run_id = mlflow.active_run().info.run_id
    print(f"✅ Model {MODEL_NAME} logged to MLflow (Run ID: {run_id})")
    print(f"🔍 View at: {MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")