import os
import numpy as np
import pandas as pd

import joblib
from pathlib import Path

from scipy.sparse import csr_matrix

class PredictionPipeline:
    """
    Prediction Pipeline for using the trained model to make predictions.

    This class provides a straightforward interface to load the trained Gradient Boosting model 
    and use it to predict on new data.

    Attributes:
    -----------
    model : object
        The trained model loaded from disk.

    Methods:
    --------
    predict(data: pd.DataFrame) -> np.array:
        Predict the target values based on input data.

    Example:
    --------
    >>> pipeline = PredictionPipeline()
    >>> new_data = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    >>> predictions = pipeline.predict(new_data)
    """

    def __init__(self):
        """
        Initializes the PredictionPipeline by loading the trained model from disk.
        """
        model_path = Path('artifacts/model_trainer/model.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")

        self.model = joblib.load(model_path)

    def predict(self, data) -> np.array:
        """
        Use the loaded model to make predictions on the input data.

        Parameters:
        -----------
        data : Union[pd.DataFrame, csr_matrix]
            The input data for which predictions are required.

        Returns:
        --------
        np.array
            The predicted values.
        """
        if not isinstance(data, (pd.DataFrame, csr_matrix)):
            raise ValueError("Input data should be a pandas DataFrame or a sparse matrix.")

        prediction = self.model.predict(data)
        return prediction
