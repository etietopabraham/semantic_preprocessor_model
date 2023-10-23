from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.config.configuration import ConfigurationManager
from src.semantic_preprocessor_model.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:

    STAGE_NAME = "Model Evaluation Pipeline"

    def __init__(self):
        self.config_manager = ConfigurationManager()

    
    def run_pipeline(self):
        try:
            logger.info("Fetching model evaluation configuration...")
            model_evaluation_configuration = self.config_manager.get_model_evaluation_config()

            logger.info("Initializing model evaluation process...")
            model_evaluation = ModelEvaluation(config=model_evaluation_configuration)
            
            logger.info("Logging model evaluation into MLFlow...")
            model_evaluation.log_into_mlflow()
            
            logger.info("Model Evaluation Pipeline completed successfully.")
       
        except Exception as e:
            logger.error(f"Error encountered during the model evaluation: {e}")


if __name__ == '__main__':
    pipeline = ModelEvaluationPipeline()
    pipeline.run_pipeline()