FROM python:3.10-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install fastapi uvicorn pandas pydantic mlflow scikit-learn scipy

CMD ["python", "main.py"]
