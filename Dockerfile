FROM python:3.10-buster

# Set working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the working directory
COPY . /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pandas pydantic mlflow scikit-learn scipy

# Command to run the training script
CMD ["python", "main.py"]
