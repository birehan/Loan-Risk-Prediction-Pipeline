import os
import logging
import mlflow
import boto3
import subprocess
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelDeployer:
    def __init__(self, model, version="v1", input_example=None):
        """
        Initialize the ModelDeployer with the model and version information.
        :param model: The trained model to be deployed.
        :param version: Version number or name for the model (default is 'v1').
        :param input_example: Example input data for model signature inference (optional).
        """
        self.model = model
        self.version = version
        self.input_example = input_example
        self.logger = logging.getLogger(__name__)

    def deploy(self, platform="local", model_name="lightgbm_model", bucket_name=None, model_path=None, region=None):
        """
        Deploy the model to the specified platform (local, MLflow remote, AWS Sagemaker, Databricks).
        :param platform: The platform to deploy to ('local', 'mlflow', 'aws', 'databricks').
        :param model_name: Name of the model to be deployed.
        :param bucket_name: Bucket name for AWS S3 storage (optional).
        :param model_path: Path to store the model locally (optional).
        :param region: AWS or Databricks region (optional).
        """
        try:
            if platform == "local":
                self.deploy_local(model_name, model_path)
            elif platform == "mlflow":
                self.deploy_mlflow_registry(model_name)
            elif platform == "aws":
                self.deploy_sagemaker(model_name, region, bucket_name)
            elif platform == "databricks":
                self.deploy_databricks(model_name, region)
            else:
                self.logger.error(f"Invalid platform '{platform}' selected.")
        except Exception as e:
            self.logger.error(f"Deployment failed on platform '{platform}' with error: {str(e)}")
            raise

    def deploy_local(self, model_name, model_path=None):
        """
        Deploy the model locally by saving it to the specified path using MLflow and register the model.
        :param model_name: The name of the model.
        :param model_path: Path where the model should be saved.
        """
        try:
            model_save_path = model_path if model_path else f"./model/{model_name}_{self.version}.txt"
            signature = infer_signature(self.input_example, self.model.predict(self.input_example))

            with mlflow.start_run() as run:
                # Log the model with signature and input example
                mlflow.lightgbm.log_model(self.model, model_name, signature=signature, input_example=self.input_example)
                mlflow.log_param("version", self.version)
                self.logger.info(f"Model saved and registered locally as {model_name} at {model_save_path}")

            # Register the model locally
            client = MlflowClient()
            client.create_registered_model(model_name)
            client.create_model_version(model_name, model_save_path, run.info.run_id)
        except MlflowException as e:
            self.logger.error(f"Failed to log model locally: {str(e)}")
            raise

    def deploy_mlflow_registry(self, model_name):
        """
        Deploy the model to a remote MLflow model registry.
        :param model_name: The name of the model to deploy.
        """
        try:
            with mlflow.start_run() as run:
                # Log model with signature and input example
                signature = infer_signature(self.input_example, self.model.predict(self.input_example))
                mlflow.lightgbm.log_model(self.model, model_name, signature=signature, input_example=self.input_example)
                mlflow.log_param("version", self.version)

                # Register the model to remote MLflow model registry
                client = MlflowClient()
                client.create_registered_model(model_name)
                client.create_model_version(model_name, f"runs:/{run.info.run_id}/model", run.info.run_id)

            self.logger.info(f"Model {model_name} registered in the MLflow remote model registry")
        except MlflowException as e:
            self.logger.error(f"Failed to deploy model to MLflow remote model registry: {str(e)}")
            raise

    def deploy_sagemaker(self, model_name, region, bucket_name):
        """
        Deploy the model to AWS Sagemaker.
        :param model_name: The name of the model.
        :param region: The AWS region for Sagemaker deployment.
        :param bucket_name: The name of the S3 bucket to upload the model artifact.
        """
        try:
            # Deploy model using MLflow to Sagemaker
            subprocess.run(
                ["mlflow", "sagemaker", "deploy", "--app-name", f"{model_name}_{self.version}",
                 "--model-uri", f"models:/{model_name}/{self.version}", "--region-name", region,
                 "--bucket-name", bucket_name],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"Model deployed to AWS Sagemaker in region {region}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to deploy model to AWS Sagemaker: {e.output}")
            raise

    def deploy_databricks(self, model_name, region):
        """
        Deploy the model to Databricks Model Serving.
        :param model_name: The name of the model.
        :param region: The Databricks workspace region.
        """
        try:
            subprocess.run(
                ["mlflow", "deployments", "create", "-t", "databricks", "--name", model_name,
                 "--model-uri", f"models:/{model_name}/{self.version}", "--workspace", region],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"Model deployed to Databricks workspace in region {region}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to deploy model to Databricks: {e.output}")
            raise

    def update_version(self, new_version):
        """
        Update the model version.
        :param new_version: New version for the model.
        """
        self.version = new_version
        self.logger.info(f"Model version updated to {self.version}")
