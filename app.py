from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from src.semantic_preprocessor_model.pipeline.predict import PredictionPipeline
from src.semantic_preprocessor_model.components.data_validation import DataValidation
from src.semantic_preprocessor_model.components.data_transformation import DataTransformation

from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from src.semantic_preprocessor_model import logger

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

app = Flask(__name__)  # Initialize Flask

app.jinja_env.globals.update(enumerate=enumerate)


@app.route('/', methods=['GET'])
def home_page():
    """
    Render the home page of the application.

    This function handles the GET request to the root URL ('/') and renders 
    the index.html template which is the home page of the application.

    Returns:
    --------
    str
        Rendered HTML template.
    """
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def train():
    os.system("python main.py")
    
    # Return a response indicating the completion of training
    return "Training Completed"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle the prediction request. It first validates the uploaded CSV file 
    and then makes predictions using the model.

    Returns:
    - Rendered HTML: Results page displaying prediction result or an appropriate error message.
    """
    logger.info("Received prediction request.")

    # Check if the post request has the file part
    if 'csvfile' not in request.files:
        logger.error("No file part in request.")
        return 'No file part'

    file = request.files['csvfile']

    if file.filename == '':
        logger.error("No file selected for upload.")
        return 'No selected file'

    if file:
        try:
            # Setup configuration manager and initialize validator
            logger.info("Setting up configuration and initializing validator.")
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            validator = DataValidation(config=data_validation_config, file_object=file)

            # Validate the uploaded file
            logger.info("Starting file validation.")
            if not validator.run_all_validations():
                logger.error("Uploaded CSV validation failed.")
                return "Uploaded CSV validation failed. Please check the file and try again."
            logger.info("File validation successful.")
            
            file.seek(0)  # Reset the file pointer to the beginning

            # Data Transformation
            logger.info("Starting data transformation.")
            data_transformation_config = config_manager.get_data_transformation_config()
            # Read the validated CSV into a DataFrame
            data_df = pd.read_csv(file)
            # Use the alternate constructor to create the DataTransformation object
            data_transformer = DataTransformation.from_dataframe(config=data_transformation_config, df=data_df)
            
            trained_vectorizer = DataTransformation.load_vectorizer("/Users/macbookpro/Documents/semantic_preprocessor_model/semantic_preprocessor_model/artifacts/vectorizersvectorizer.pkl")

            data_transformer.vectorizer_work_name = trained_vectorizer

            X_test, y_test = data_transformer.transform_test_data(data_df)
            logger.info("Data transformation successful.")

            # Make the prediction using the prediction pipeline
            logger.info("Starting prediction.")
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(X_test)
            logger.info("Prediction successful.")

            # Evaluate the predictions
            logger.info("Starting evaluation.")
            precision_general, recall_general, f1_general, _ = precision_recall_fscore_support(y_test, prediction, average='weighted')
            accuracy_general = accuracy_score(y_test, prediction)
            logger.info("Evaluation successful.")

            # Limit the prediction to top 20
            limited_predictions = prediction.tolist()[:50]

            # Create a result object
            result = {
                "prediction": limited_predictions,
                "precision": precision_general,
                "recall": recall_general,
                "f1_score": f1_general,
                "accuracy": accuracy_general
            }

            logger.info("Result object created.")

            # Render and return the results to results page
            return render_template('results.html', result_object=result)

        except Exception as e:
            # Log the exception for debugging
            logger.error(f"Error occurred during prediction: {e}")

            # Return a user-friendly error message
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8484, debug=True)