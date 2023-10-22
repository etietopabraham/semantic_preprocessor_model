from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.components.model_training import ModelTraining
from src.semantic_preprocessor_model.config.configuration import ConfigurationManager

class ModelTrainerPipeline:
    """
    This pipeline handles the model training process.

    After the data transformation stage, this class orchestrates the training of the model
    using the GradientBoostingRegressor and saves the trained model for future use.

    Attributes:
        STAGE_NAME (str): The name of this pipeline stage.
    """
    
    STAGE_NAME = "Model Training Pipeline"

    def __init__(self):
        """
        Initializes the pipeline with a configuration manager.
        """
        self.config_manager = ConfigurationManager()

    def run_model_training(self):
        """
        Orchestrates the model training process.

        Fetches configurations, initializes the model training process, trains the model,
        and logs the successful completion of the training.
        """
        try:
            logger.info("Fetching model training configuration...")
            model_training_configuration = self.config_manager.get_model_training_config()

            logger.info("Initializing model training process...")
            model_training = ModelTraining(config=model_training_configuration)

            logger.info("Executing model training...")
            model_training.train()

            logger.info("Model Training Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error encountered during the model training: {e}")

    
    def run_pipeline(self):
        """
        Run the entire Model Training Pipeline.

        This method orchestrates the process of model training and provides logs for each stage 
        of the pipeline.
        """
        try:
            logger.info("Starting the Model Training Pipeline.")
            logger.info(f">>>>>> Stage: {ModelTrainerPipeline.STAGE_NAME} started <<<<<<")
            self.run_model_training()
            logger.info(f">>>>>> Stage {ModelTrainerPipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
        except Exception as e:
            logger.error(f"Error encountered during the {ModelTrainerPipeline.STAGE_NAME}: {e}")
            raise e


if __name__ == '__main__':
    pipeline = ModelTrainerPipeline()
    pipeline.run_pipeline()