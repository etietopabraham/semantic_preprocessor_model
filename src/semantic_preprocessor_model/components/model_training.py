from sklearn.neural_network import MLPClassifier
from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from scipy.sparse import load_npz
import joblib
import os
import pandas as pd
import ast

from src.semantic_preprocessor_model import logger


class ModelTraining:
    """
    ModelTraining is responsible for training a machine learning model based on the 
    provided configuration. It uses the MLPClassifier to train a neural network model 
    on the processed and transformed data.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initializes the ModelTraining component.
        
        Args:
        - config (ConfigurationManager): Configuration settings for model training.
        """
        self.config = config

    def train(self):
        """
        Train the model using the transformed data. The method loads the training data, 
        initializes the MLPClassifier with the specified parameters, trains the classifier, 
        and then saves the trained model to the specified path.
        """
        
        # Load training data
        X_train = load_npz(self.config.train_features_path)
        y_train = pd.read_csv(self.config.train_labels_path).iloc[:, 0]

        # Convert string representation of tuple to actual tuple
        hidden_layer_sizes_tuple = ast.literal_eval(self.config.hidden_layer_sizes)

        params = {
            'hidden_layer_sizes': hidden_layer_sizes_tuple,
            'max_iter': self.config.max_iter,
            'random_state': self.config.random_state
        }

        # Initialize a Neural Network classifier with the specified parameters
        nn_classifier_general = MLPClassifier(**params, verbose=True)

        print(X_train.shape[0])
        print(len(y_train))

        print(self.config.hidden_layer_sizes)
        print(type(self.config.hidden_layer_sizes))

        # Train the Neural Network classifier
        nn_classifier_general.fit(X_train, y_train)

        # Save the trained model
        model_save_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(nn_classifier_general, model_save_path)
        logger.info(f"Model saved successfully to {model_save_path}")

