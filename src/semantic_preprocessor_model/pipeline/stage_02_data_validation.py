from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from src.semantic_preprocessor_model.components.data_validation import DataValidation

class DataValidationPipeline:
    """
    This pipeline handles the initial data validation steps.

    After the data ingestion stage, it's imperative to ensure the data's integrity
    before moving on to feature engineering or model training. This class
    orchestrates that validation by checking for correct features and data types.

    Attributes:
        STAGE_NAME (str): The name of this pipeline stage.
    """

    STAGE_NAME = "Data Validation Pipeline"

    def __init__(self):
        """
        Initializes the pipeline with a configuration manager.
        """
        self.config_manager = ConfigurationManager()

    def run_data_validation(self):
        """
        Run the set of data validations.
        
        This method orchestrates the different validation functions to ensure the
        dataset's integrity.
        """
        try:
            logger.info("Fetching initial data validation configuration...")
            data_validation_config = self.config_manager.get_data_validation_config()

            logger.info("Initializing data validation process...")
            data_validation = DataValidation(config=data_validation_config)

            logger.info("Executing Data Validations...")
            data_validation.run_all_validations()

            logger.info("Initial Data Validation Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error encountered during the data validation: {e}")
    
    def run_pipeline(self):
        """
        Run the entire Initial Data Validation Pipeline.
        
        This method encapsulates the process of the initial data validation and
        provides logs for each stage of the pipeline.
        """
        try:
            logger.info(f">>>>>> Stage: {DataValidationPipeline.STAGE_NAME} started <<<<<<")
            self.run_data_validation()
            logger.info(f">>>>>> Stage {DataValidationPipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
        except Exception as e:
            logger.error(f"Error encountered during the {DataValidationPipeline.STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    pipeline = DataValidationPipeline()
    pipeline.run_pipeline()
