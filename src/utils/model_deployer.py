# import os
# import logging
# import mlflow
# import boto3
# from google.cloud import storage
# from mlflow.exceptions import MlflowException
# import subprocess
# from kubernetes import client, config

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# class ModelDeployer:
#     def __init__(self, model, version="v1"):
#         """
#         Initialize the ModelDeployer with the model and version information.
#         :param model: The trained model to be deployed.
#         :param version: Version number or name for the model (default is 'v1').
#         """
#         self.model = model
#         self.version = version
#         self.logger = logging.getLogger(__name__)

#     def deploy(self, platform="local", model_name="lightgbm_model", bucket_name=None, model_path=None, region=None):
#         """
#         Deploy the model to the specified platform (local, AWS Sagemaker, Databricks, GCS).
#         :param platform: The platform to deploy to ('local', 'aws', 'databricks', 'gcs', 'kubernetes', 'mlflow', 'sagemaker').
#         :param model_name: Name of the model to be deployed.
#         :param bucket_name: Bucket name for cloud storage (AWS S3, GCS).
#         :param model_path: The path to store the model locally or in cloud storage.
#         :param region: The AWS region for Sagemaker or Databricks workspace region.
#         """
#         try:
#             if platform == "local":
#                 self.deploy_local(model_name, model_path)
#             elif platform == "aws":
#                 self.deploy_aws(model_name, bucket_name, model_path)
#             elif platform == "databricks":
#                 self.deploy_databricks(model_name, region)
#             elif platform == "gcs":
#                 self.deploy_gcs(model_name, bucket_name, model_path)
#             elif platform == "kubernetes":
#                 self.deploy_kubernetes(model_name)
#             elif platform == "mlflow":
#                 self.deploy_mlflow(model_name)
#             elif platform == "sagemaker":
#                 self.deploy_sagemaker(model_name, region)
#             else:
#                 self.logger.error(f"Invalid platform '{platform}' selected.")
#         except Exception as e:
#             self.logger.error(f"Deployment failed on platform '{platform}' with error: {str(e)}")
#             raise

#     def deploy_local(self, model_name, model_path=None):
#         """
#         Deploy the model locally by saving it to the specified path.
#         :param model_name: The name of the model.
#         :param model_path: Path where the model should be saved.
#         """
#         try:
#             model_save_path = model_path if model_path else f"./model/{model_name}_{self.version}.txt"
#             self.model.booster_.save_model(model_save_path)
#             self.logger.info(f"Model saved locally at {model_save_path}")
#         except Exception as e:
#             self.logger.error(f"Failed to save the model locally: {str(e)}")
#             raise

#     def deploy_aws(self, model_name, bucket_name, model_path=None):
#         """
#         Deploy the model to AWS S3.
#         :param model_name: The name of the model.
#         :param bucket_name: The name of the S3 bucket.
#         :param model_path: Local path to temporarily store the model before uploading.
#         """
#         try:
#             s3_client = boto3.client('s3')
#             model_save_path = model_path if model_path else f"./model/{model_name}_{self.version}.txt"
#             self.model.booster_.save_model(model_save_path)
#             s3_client.upload_file(model_save_path, bucket_name, os.path.basename(model_save_path))
#             self.logger.info(f"Model uploaded to AWS S3 bucket '{bucket_name}' as {model_name}_{self.version}")
#         except Exception as e:
#             self.logger.error(f"Failed to upload model to AWS S3: {str(e)}")
#             raise

#     def deploy_sagemaker(self, model_name, region):
#         """
#         Deploy the model to AWS Sagemaker.
#         :param model_name: The name of the model.
#         :param region: The AWS region for Sagemaker deployment.
#         """
#         try:
#             # Deploy model using MLflow to Sagemaker
#             subprocess.run(
#                 ["mlflow", "sagemaker", "deploy", "--app-name", f"{model_name}_{self.version}",
#                  "--model-uri", f"models:/{model_name}/{self.version}", "--region-name", region],
#                 check=True,
#                 capture_output=True,
#                 text=True
#             )
#             self.logger.info(f"Model deployed to AWS Sagemaker in region {region}")
#         except subprocess.CalledProcessError as e:
#             self.logger.error(f"Failed to deploy model to AWS Sagemaker: {e.output}")
#             raise

#     def deploy_databricks(self, model_name, region):
#         """
#         Deploy the model to Databricks Model Serving.
#         :param model_name: The name of the model.
#         :param region: The Databricks workspace region.
#         """
#         try:
#             subprocess.run(
#                 ["mlflow", "deployments", "create", "-t", "databricks", "--name", model_name,
#                  "--model-uri", f"models:/{model_name}/{self.version}", "--workspace", region],
#                 check=True,
#                 capture_output=True,
#                 text=True
#             )
#             self.logger.info(f"Model deployed to Databricks workspace in region {region}")
#         except subprocess.CalledProcessError as e:
#             self.logger.error(f"Failed to deploy model to Databricks: {e.output}")
#             raise

#     def deploy_gcs(self, model_name, bucket_name, model_path=None):
#         """
#         Deploy the model to Google Cloud Storage (GCS).
#         :param model_name: The name of the model.
#         :param bucket_name: The name of the GCS bucket.
#         :param model_path: Local path to temporarily store the model before uploading.
#         """
#         try:
#             client = storage.Client()
#             model_save_path = model_path if model_path else f"./model/{model_name}_{self.version}.txt"
#             self.model.booster_.save_model(model_save_path)
#             bucket = client.bucket(bucket_name)
#             blob = bucket.blob(os.path.basename(model_save_path))
#             blob.upload_from_filename(model_save_path)
#             self.logger.info(f"Model uploaded to GCS bucket '{bucket_name}' as {model_name}_{self.version}")
#         except Exception as e:
#             self.logger.error(f"Failed to upload model to GCS: {str(e)}")
#             raise

#     def deploy_mlflow(self, model_name):
#         """
#         Deploy the model using MLflow for model tracking, serving, and versioning.
#         :param model_name: The name of the model.
#         """
#         try:
#             with mlflow.start_run():
#                 mlflow.lightgbm.log_model(self.model, model_name)
#                 mlflow.log_param("version", self.version)
#                 self.logger.info(f"Model logged to MLflow as {model_name} with version {self.version}")
#         except MlflowException as e:
#             self.logger.error(f"Failed to log model to MLflow: {str(e)}")
#             raise

#     def deploy_kubernetes(self, model_name):
#         """
#         Deploy the model to a Kubernetes Cluster.
#         :param model_name: The name of the model.
#         """
#         try:
#             config.load_kube_config()
#             v1 = client.CoreV1Api()
#             self.logger.info("Listing Kubernetes nodes:")
#             print("Nodes in the cluster:")
#             nodes = v1.list_node()
#             for node in nodes.items:
#                 print(f"Node: {node.metadata.name}")
#             self.logger.info(f"Deploying {model_name} to Kubernetes...")
#             # You can also integrate the deployment of Kubernetes resources using the Kubernetes Python client.
#         except Exception as e:
#             self.logger.error(f"Failed to deploy to Kubernetes: {str(e)}")
#             raise

#     def update_version(self, new_version):
#         """
#         Update the model version.
#         :param new_version: New version for the model.
#         """
#         self.version = new_version
#         self.logger.info(f"Model version updated to {self.version}")



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
