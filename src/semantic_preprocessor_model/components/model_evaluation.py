from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pathlib import Path
from src.semantic_preprocessor_model import logger

from src.semantic_preprocessor_model.config.configuration import ConfigurationManager


import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import joblib
import mlflow
from scipy.sparse import load_npz
import ast

from src.semantic_preprocessor_model.utils.common import save_json

class ModelEvaluation:
    """
    The ModelEvaluation class evaluates the performance of a trained model using 
    validation data and logs the results into MLflow.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize ModelEvaluation with a configuration manager.

        Args:
        - config (ConfigurationManager): Configuration manager instance.
        """
        self.config = config
        self.X_val = None
        self.y_val = None
        self.model = None

    def eval_metrics(self, actual, pred):
        """
        Calculate evaluation metrics for classification.
        
        Args:
        - actual (array-like): True labels.
        - pred (array-like): Predicted labels.
        
        Returns:
        - dict: Dictionary containing accuracy, precision, recall, and F1 score.
        """
        accuracy = accuracy_score(actual, pred)
        
        # Calculate precision, recall, and F1
        precision_values, recall_values, f1_values, _ = precision_recall_fscore_support(actual, pred, average='weighted')
        
        # Take average for logging purposes
        precision = precision_values
        recall = recall_values
        f1 = f1_values
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
        print(f"Accuracy type: {type(accuracy)}")
        print(f"All Params: {self.config.all_params}")
        print(f"Metrics {results}")
    
        return results
    
    def load_data(self):
        """
        Load validation data and the trained model.
        """
        try:
            # Load validation data
            logger.info("Loading validation features...")
            self.X_val = load_npz(self.config.val_features_path)

            logger.info("Loading validation labels...")
            self.y_val = pd.read_csv(self.config.val_labels_path).iloc[:, 0]

            logger.info("Loading trained model...")
            self.model = joblib.load(self.config.model_path)

            logger.info("Data and model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error while loading data or model: {e}")
            raise e

    def log_into_mlflow(self):
        """
        Log model parameters, metrics, and the model itself into MLflow. This function first loads 
        the validation data and the trained model. It then predicts on the validation data using 
        the model and calculates evaluation metrics. These metrics, along with model parameters, 
        are then logged into MLflow. Finally, the model itself is also logged into MLflow.
        """
        # Logging the start of the MLflow logging process
        logger.info("Starting MLflow logging...")

        # Load validation data and the trained model
        self.load_data()

        # Set the MLflow registry URI
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme

        # If the model configuration contains 'hidden_layer_sizes', convert its string representation to actual tuple
        if 'hidden_layer_sizes' in self.config.all_params:
            self.config.all_params['hidden_layer_sizes'] = ast.literal_eval(self.config.all_params['hidden_layer_sizes'])

        # Start an MLflow tracking session
        with mlflow.start_run():
            # Predict on validation data
            predicted_qualities = self.model.predict(self.X_val)

            # Calculate evaluation metrics
            metrics = self.eval_metrics(self.y_val, predicted_qualities)
            scores = {
                "accuracy": metrics['accuracy'], 
                "precision": metrics['precision'], 
                "recall": metrics['recall'], 
                "f1": metrics['f1']
            }

            # Save the calculated metrics to a JSON file
            save_json(path=Path(self.config.metric_file_path), data=scores)

            # Log model parameters into MLflow
            mlflow.log_params(self.config.all_params)

            # Log each metric into MLflow
            for key, value in scores.items():
                mlflow.log_metric(key, value)

            # Determine how to log the model into MLflow based on the tracking URL type
            if tracking_url_type_score != "file":
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="MLPClassifier")
            else:
                mlflow.sklearn.log_model(self.model, "model")

        # Logging the completion of the MLflow logging process
        logger.info("MLflow logging completed.")
