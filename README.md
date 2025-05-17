# Install packages

pip install -r requirements.txt

# Build and Run docker image

docker build -t model_trainer . && docker run -d --name model_trainer --network mlflow-net -v "D:/mlflow/mlruns:/mlflow/mlruns" model_trainer
