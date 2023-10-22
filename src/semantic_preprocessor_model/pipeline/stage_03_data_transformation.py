from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from src.semantic_preprocessor_model.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    """
    Orchestrates data transformation processes:
    - Text preprocessing
    - Missing value handling
    - Text feature vectorization
    - Data filtering and splitting
    - Saving transformed datasets
    """

    STAGE_NAME = "Data Transformation Pipeline"

    def __init__(self):
        """Initialize the pipeline with a configuration manager."""
        self.config_manager = ConfigurationManager()

    def run_data_transformation(self):
        """
        Execute data transformation steps and log each stage.

        Raises:
            Exception: If any error occurs during the data transformation process.
        """
        try:
            logger.info("Fetching data transformation configuration...")
            data_transformation_config = self.config_manager.get_data_transformation_config()

            logger.info("Initializing data transformation...")
            data_transformer = DataTransformation(config=data_transformation_config)

            logger.info("Executing data transformation pipeline...")
            X_train, X_val, y_train, y_val = data_transformer.transform()
            
            logger.info(f"Shape of X_train: {X_train.shape}")
            logger.info(f"Shape of X_val: {X_val.shape}")
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_val: {y_val.shape}")

            logger.info("Data Transformation Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")

    def run_pipeline(self):
        """
        Run the entire Data Transformation Pipeline, checking data validations before proceeding.

        Raises:
            Exception: If any error occurs during the pipeline execution.
        """
        try:
            with open(self.config_manager.get_data_transformation_config().data_validation, "r") as f:
                content = f.read()

            # Ensure the validations have passed before running the pipeline
            if "Overall Validation Status: All validations passed." in content:
                logger.info("Starting the Data Transformation Pipeline.")
                logger.info(f">>>>>> Stage: {DataTransformationPipeline.STAGE_NAME} started <<<<<<")
                self.run_data_transformation()
                logger.info(f">>>>>> Stage: {DataTransformationPipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
            else:
                logger.error("Pipeline aborted due to validation errors.")

        except Exception as e:
            logger.error(f"Error during {DataTransformationPipeline.STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    pipeline = DataTransformationPipeline()
    pipeline.run_pipeline()