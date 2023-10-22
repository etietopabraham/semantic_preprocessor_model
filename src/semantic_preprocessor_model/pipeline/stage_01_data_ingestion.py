from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from src.semantic_preprocessor_model.components.data_ingestion import DataIngestion
from src.semantic_preprocessor_model import logger

class DataIngestionPipeline:

    STAGE_NAME = "Data Ingestion Stage"

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_data_ingestion(self):
        """
        Main method to run the data ingestion process.
        """
        try:
            logger.info("Fetching data ingestion configuration...")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            
            logger.info("Initializing data ingestion process...")
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            logger.info(f"Copying training data from {data_ingestion_config.local_data_file} to {data_ingestion_config.root_dir}...")
            data_ingestion.transfer_data()
            
        except Exception as e:
            logger.exception("An error occurred during the data ingestion process.")
            raise e
        
    def run_pipeline(self):
        """
        Run the data ingestion training pipeline.
        """
        try:
            logger.info(f">>>>>> Stage: {DataIngestionPipeline.STAGE_NAME} started <<<<<<")
            self.run_data_ingestion()
            logger.info(f">>>>>> Stage {DataIngestionPipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
        except Exception as e:
            # No need to log the exception here since it's already logged in the run_data_ingestion method.
            raise e

if __name__ == '__main__':
    pipeline = DataIngestionPipeline()
    pipeline.run_pipeline()