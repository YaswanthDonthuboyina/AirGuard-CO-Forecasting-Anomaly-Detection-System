# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# 1. Prevents Python from writing .pyc files to disc
# 2. Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to keep the image size smaller
# --upgrade: Ensures pip is updated
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the local directories to the container
# Copy source code for preprocessing, feature engineering, and prediction
COPY ./src /app/src
# Copy the API application code
COPY ./api /app/api
# Copy the trained model, scaler, and OOD detector
COPY ./models /app/models


# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# uvicorn will start the FastAPI application
# --host 0.0.0.0 makes the app accessible from outside the container
# --port 8000 specifies the port to run on
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
