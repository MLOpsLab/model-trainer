# Use official Python base image
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install necessary dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn pandas pydantic mlflow scikit-learn

# Command to run the training script
CMD ["python", "main.py"]
