FROM python:3.9-slim

WORKDIR /app

# System dependencies for scikit-learn, pandas, MLflow
RUN apt-get update && \
    apt-get install -y build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY main.py /app/model_trainer.py

RUN pip install --no-cache-dir \
    pandas scikit-learn mlflow

CMD ["python", "model_trainer.py"]