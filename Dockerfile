FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
 && pip install --prefer-binary mlflow scikit-learn pandas scipy

CMD ["python", "main.py"]
