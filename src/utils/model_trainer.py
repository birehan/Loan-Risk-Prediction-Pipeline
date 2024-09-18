
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flaml import AutoML
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import logging
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import os
import json

from src.utils.utils import determine_version, update_version_file

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LightGBMAutoML:
    def __init__(self, X_train, X_test, y_train, y_test, init_model=None):
        """
        Initialize the AutoML model with pre-split data.
        :param X_train: Training feature set.
        :param X_test: Testing feature set.
        :param y_train: Training target set.
        :param y_test: Testing target set.
        :param init_model: Initial AutoML model (optional), used for versioning.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.automl = AutoML()
        self.model = None
        self.version = determine_version()
        self.run_id = None



    def print_auto_logged_info(self, run):
        """
        Print information about the auto-logged MLflow run.
        :param run: The MLflow run object.
        """
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [
            f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, "model")
        ]
        feature_importances = [
            f.path
            for f in mlflow.MlflowClient().list_artifacts(run.info.run_id)
            if f.path != "model"
        ]
        print(f"run_id: {run.info.run_id}")
        print(f"artifacts: {artifacts}")
        print(f"feature_importances: {feature_importances}")
        print(f"params: {run.data.params}")
        print(f"metrics: {run.data.metrics}")
        print(f"tags: {tags}")

    def train_model(self, time_budget=240):
        """
        Train the AutoML model using the training data for multi-class classification.
        :param time_budget: Total time budget for the training in seconds.
        """
        logging.info("Starting AutoML training for multi-class classification...")        
        
        # Enable MLflow autologging
        mlflow.lightgbm.autolog()

        # AutoML settings for multi-class classification
        settings = {
            "time_budget": time_budget,
            "metric": "accuracy",
            "estimator_list": ["lgbm"],
            "task": "classification",
            "log_file_name": "multi_class_experiment.log",
            "seed": 7654321,
        }

        # Specify categorical features
        categorical_features = [col for col in self.X_train.columns if self.X_train[col].dtype == 'category']

        with mlflow.start_run() as run:
            # Fit the AutoML model
            self.automl.fit(X_train=self.X_train, y_train=self.y_train, 
                            categorical_feature=categorical_features, **settings)
            self.model = self.automl.model

            # Print auto-logged information
            self.print_auto_logged_info(run)
            
            self.run_id = run.info.run_id

            # Log accuracy
            y_pred = self.automl.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            #
            input_example = self.X_test.iloc[:1]
            signature = infer_signature(input_example, self.automl.predict(input_example))

            # Register the model with versioning
            mlflow.lightgbm.log_model(self.automl, "model", signature=signature, input_example=input_example)
            mlflow.log_param("version", self.version)  # Log the version
            
            logging.info(f"Best model found: {self.automl.best_estimator}")
            logging.info(f"Model version {self.version} registered and logged.")
            

            # Update version file after successful training
            update_version_file(self.version)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        y_pred = self.automl.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification report:\n{report}")
        return accuracy, report

    def plot_feature_importance(self) -> None:
        """
        Plot feature importance based on the best model.
        """
        feature_importance = self.model.feature_importances_
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_metrics(self) -> None:
        """
        Plot metrics recorded during training.
        """
        if hasattr(self.model, 'best_iteration_'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.evals_result_['training']['binary_logloss'], label='Training Loss')
            plt.plot(self.model.evals_result_['valid']['binary_logloss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("No metrics available to plot.")

    def plot_split_value_histogram(self, feature: str) -> None:
        """
        Plot a histogram of the split values for the best model.
        """
        if self.model:
            plt.figure(figsize=(10, 6))
            lgb.plot_split_value_histogram(self.model.model, feature)
            plt.title('Split Value Histogram')
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("Model not trained yet; cannot plot split value histogram.")

    def plot_trees(self, tree_index:int=0) -> None:
        """
        Plot a tree from the trained model.
        :param tree_index: Index of the tree to plot.
        """
        if self.model:
            plt.figure(figsize=(20, 10))
            lgb.plot_tree(self.model.model,figsize=(20, 10),tree_index=tree_index)
            plt.title(f'Tree {tree_index}')
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("Model not trained yet; cannot plot tree.")

# # Example usage:
# if __name__ == "__main__":
#     # Configure MLflow tracking URI (if needed)
#     # mlflow.set_tracking_uri("http://your_mlflow_server:5000")  # Uncomment and set if using a remote server

#     # Assuming X_train, X_test, y_train, y_test are already defined
#     # Example: 
#     # from sklearn.model_selection import train_test_split
#     # df = pd.read_csv('your_preprocessed_data.csv')
#     # X = df.drop(columns=['target_column'])
#     # y = df['target_column']
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create an instance of the LightGBMAutoML class
#     lightgbm_automl = LightGBMAutoML(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

#     # Train the model with AutoML
#     lightgbm_automl.train_model(time_budget=300, n_trials=20)  # Adjust time_budget and n_trials as needed

#     # Evaluate the model
#     lightgbm_automl.evaluate_model()

#     # Plot feature importance
#     lightgbm_automl.plot_feature_importance()

#     # Plot training metrics
#     lightgbm_automl.plot_metrics()

#     # Plot split value histogram
#     lightgbm_automl.plot_split_value_histogram()

#     # Plot a specific tree
#     lightgbm_automl.plot_trees(tree_index=0)  # Change tree_index to visualize different trees
