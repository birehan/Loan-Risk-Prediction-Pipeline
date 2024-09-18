# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copy the requirements file from the specified model path
ARG MODEL_PATH
COPY ${MODEL_PATH}/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files into the container
COPY ${MODEL_PATH} .

# Expose the port that MLflow will run on
EXPOSE 5000

# Set the entry point to run the MLflow server and bind it to 0.0.0.0
CMD ["mlflow", "models", "serve", "-m", ".", "-h", "0.0.0.0", "-p", "5000", "--no-conda"]
