# Loan Risk Prediction Machine Learning Pipeline

This repository contains a machine learning pipeline designed to predict the risk of loan applications using multiple datasets. The pipeline covers the complete workflow from data ingestion to model deployment, leveraging LightGBM, FLAML, MLflow, Docker, and cloud platforms for scalable and efficient model management.

## Table of Contents
- [Overview](#overview)
- [Pipeline Components](#pipeline-components)
  - [Data Preparation](#data-preparation)
  - [Modeling](#modeling)
  - [Deployment](#deployment)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Serving the Model as an API](#serving-the-model-as-an-api)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
- [Model Retraining and Deployment](#model-retraining-and-deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview
This machine learning pipeline predicts loan application risk based on various factors using the following datasets:
- `loan.csv`: Loan details such as amount, status, and application date.
- `payment.csv`: Payment history.
- `clarity_underwriting_variables.csv`: Underwriting and risk variables.

The model is built with LightGBM, and hyperparameter tuning is automated using FLAML AutoML. The pipeline integrates with MLflow for model tracking and logging and leverages Docker for containerization, with options for deployment on AWS SageMaker, Databricks, Google Cloud, or Render.com.

## Pipeline Components

### 1. Data Preparation
- **Data Ingestion**: Loan data is ingested from CSV files. Future integrations with databases and APIs can be added for expansion.
- **Data Validation**: The dataset is validated to ensure it conforms to the schema defined in `schema.json`.
- **Data Pre-processing**:
  - Handling missing values by dropping columns or rows with excessive missing values or filling them with mean/median/mode.
  - Removing highly correlated columns, dropping duplicates, and converting data types.
  - Splitting data into training and testing sets (80/20).

### 2. Modeling
- **Hyperparameter Tuning**: FLAML AutoML optimizes LightGBM hyperparameters, ensuring a faster and more efficient approach.
- **Model Training & Tuning**: The model is trained using the optimized hyperparameters, and training metrics such as accuracy and loss are logged for continuous monitoring.
- **Model Evaluation & Analysis**: Performance is evaluated using accuracy, precision, and recall. Feature importance is visualized, and MLflow logs and tracks model versions, hyperparameters, and metrics.

### 3. Deployment
- **Model Versioning**: Each iteration of the model is versioned for tracking updates.
- **Containerization**: The model and its dependencies are packaged into a Docker container for consistent deployment across environments.
- **Model Serving**: The Docker image is deployed on Render.com, with options for AWS SageMaker, Databricks, or Google Cloud. Predictions are served via an API.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Docker
- MLflow
- FLAML
- Streamlit
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/birehan/Loan-Risk-Prediction-Pipeline.git
   cd loan-risk-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MLflow for model tracking:
   ```bash
   mlflow ui
   ```

### Running the Pipeline
To run the complete pipeline:
```bash
python main.py
```

This will ingest data, preprocess it, train the model with hyperparameter tuning using FLAML, evaluate the model, and store the results in MLflow.

### Serving the Model as an API
After training the model, you can serve it using MLflow as a REST API:
1. Run the following command to serve the model:
   ```bash
   mlflow models serve -m models:/loan-risk-model/production -p 5000
   ```
   This will start an API service on port 5000, serving the model stored in the MLflow registry under the name `loan-risk-model`.

2. You can now send requests to the API endpoint. For example, using `curl`:
   ```bash
   curl -X POST http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
       "columns": ["feature1", "feature2", "feature3"],
       "data": [[value1, value2, value3]]
   }'
   ```

### Running the Streamlit Dashboard
A Streamlit dashboard is provided to visualize model predictions and key metrics:
1. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

2. This will open the Streamlit dashboard in your web browser. You can interact with the model and visualize predictions directly through the UI.

## Model Retraining and Deployment
The model undergoes monthly retraining to handle data drift. GitHub Actions is used to automate the process:
- **Monthly Retraining**: Automatically retrains the model using fresh data.
- **AutoML Tuning**: FLAML optimizes the hyperparameters during each retraining cycle.
- **Container Deployment**: A Docker container is built and pushed to the specified deployment platform (e.g., AWS SageMaker, Render.com) after retraining.

To trigger deployment manually:
```bash
docker build -t loan-risk-prediction .
docker push your-docker-repo/loan-risk-prediction
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

## License
This project is licensed under the MIT License.