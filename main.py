from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.semantic_preprocessor_model.pipeline.stage_02_data_validation import DataValidationPipeline
from src.semantic_preprocessor_model.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.semantic_preprocessor_model.pipeline.stage_04_data_training import ModelTrainerPipeline
# from src.predicting_publications.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

def main():
    """
    Main orchestrator function to execute all the pipeline stages in the defined sequence.
    
    The function loops through each stage in the execution sequence, initiates, and runs it.
    Any errors encountered during a stage's execution are logged, and the program is terminated.
    """
    
    # Define the list of pipeline stages to be executed in sequence
    execution_sequence = [DataIngestionPipeline(), 
                          DataValidationPipeline(),
                          DataTransformationPipeline(),
                          ModelTrainerPipeline(),
                        #   ModelEvaluationPipeline()
                          ]

    for pipeline in execution_sequence:
        try:
            # Start and log the current pipeline stage
            logger.info(f">>>>>> Stage: {pipeline.STAGE_NAME} started <<<<<<")
            
            # Execute the `run_pipeline` method of the current pipeline
            pipeline.run_pipeline()
            
            # Log the successful completion of the current pipeline stage
            logger.info(f">>>>>> Stage {pipeline.STAGE_NAME} completed <<<<<< \n\nx==========x")
        except Exception as e:
            # Log any errors encountered during the pipeline's execution
            logger.exception(f"Error encountered during the {pipeline.STAGE_NAME}: {e}")
            logger.error("Program terminated due to an error.")
            
            # Exit the program with an error status
            exit(1)

if __name__ == "__main__":
    # Start the main orchestrator function if the script is run as the main module
    main()