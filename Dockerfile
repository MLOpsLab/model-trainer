# Use official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training code from GitHub repo into the container
COPY . /app

# Command to run the training script
CMD ["python", "main.py"]
