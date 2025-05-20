# Use official Python base image
FROM python:3.10-slim

# Install necessary dependencies
RUN pip install fastapi uvicorn pandas pydantic mlflow scikit-learn

# Command to run the training script
CMD ["python", "main.py"]
