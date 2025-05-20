# Use official Python base image
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Set environment variables to avoid warnings during installation
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

# Update pip first to avoid errors with older versions
RUN pip install --upgrade pip

# Install necessary dependencies one by one to debug easily
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pandas
RUN pip install pydantic
RUN pip install mlflow
RUN pip install scikit-learn

# Command to run the training script
CMD ["python", "main.py"]
