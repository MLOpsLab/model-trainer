import boto3
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# S3 configuration
S3_BUCKET = os.getenv('ARTIFACT_URI')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/random_forest.joblib')
DATASET_URI = os.getenv('DATASET_URI')

# Train model
df = pd.read_csv(DATASET_URI)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model locally first
local_path = '/tmp/model.joblib'
joblib.dump(model, local_path)

# Upload to S3
s3_client = boto3.client('s3')
s3_client.upload_file(local_path, S3_BUCKET, MODEL_PATH)
print(f"✅ Model successfully trained and uploaded to s3://{S3_BUCKET}/{MODEL_PATH}")
