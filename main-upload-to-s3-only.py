import os
import pandas as pd
import joblib
import boto3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration from environment variables with validation
S3_BUCKET = os.getenv('S3_BUCKET')
S3_KEY = os.getenv('S3_KEY', 'models/diabetes_model.joblib')
DATASET_URI = os.getenv('DATASET_URI')

# Validate required parameters
if not S3_BUCKET:
    raise ValueError("S3_BUCKET environment variable is not set")
if not DATASET_URI:
    raise ValueError("DATASET_URI environment variable is not set")

print(f"Starting model training with dataset: {DATASET_URI}")
print(f"Will upload to: s3://{S3_BUCKET}/{S3_KEY}")

# Load and prepare data
df = pd.read_csv(DATASET_URI)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save model locally
local_path = '/tmp/model.joblib'
joblib.dump(model, local_path)
print(f"Model saved locally to: {local_path}")

# Debug information for AWS setup
print("AWS Configuration:")
print(f"  S3_BUCKET: {S3_BUCKET}")
print(f"  S3_KEY: {S3_KEY}")
print(f"  AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")
print(f"  AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not set'}")
print(f"  AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not set'}")

# Upload to S3
try:
    print(f"Uploading model to S3...")
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        Filename=local_path,
        Bucket=S3_BUCKET,
        Key=S3_KEY
    )
    print(f"✅ SUCCESS: Model uploaded to s3://{S3_BUCKET}/{S3_KEY}")
except Exception as e:
    print(f"❌ ERROR: Failed to upload model to S3: {str(e)}")
    print("Full error details:")
    import traceback
    traceback.print_exc()
finally:
    # Clean up
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"Cleaned up local file: {local_path}")