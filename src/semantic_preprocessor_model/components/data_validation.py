import pandas as pd
from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    The DataValidation class ensures the integrity of the dataset by comparing it 
    against a predefined schema. It verifies the presence and data types of columns 
    as per the expectations set in the schema.

    Attributes:
    - df (pd.DataFrame): The data to be validated.
    """

    # Define optional and required columns for validation
    optional_columns = {'upper_work'}
    required_columns = {'work_name', 'generalized_work_class', 'global_work_class'}

    def __init__(self, config: DataValidationConfig, file_object=None):
        """
        Initializes the DataValidation class.
        
        Depending on the presence of a file_object, it either loads data from the provided 
        file object or from the specified file in the configuration.

        Args:
        - config (DataValidationConfig): Configuration settings for data validation.
        - file_object (File, optional): A file object containing the dataset.
        """
        logger.info("Initializing DataValidation.")
        self.config = config
        try:
            if file_object:
                self.df = pd.read_csv(file_object)
            else:
                self.df = pd.read_csv(self.config.data_source_file)
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_source_file}")
            raise

    def validate_all_features(self) -> bool:
        """
        Checks if all expected columns, as defined in the schema, are present in the dataframe.

        Returns:
        - bool: True if all columns are present and match the schema, False otherwise.
        """
        logger.info("Starting feature validation.")
        
        validation_status = True

        all_columns = set(self.df.columns)
        expected_columns = set(self.config.schema.keys())

        missing_required_columns = self.required_columns - all_columns
        extra_columns = all_columns - expected_columns - self.optional_columns

        if missing_required_columns:
            validation_status = False
            logger.warning(f"Missing required columns: {', '.join(missing_required_columns)}")

        if extra_columns:
            validation_status = False
            logger.warning(f"Extra columns found: {', '.join(extra_columns)}")

        if validation_status:
            logger.info("All expected columns are present in the dataframe.")
        return validation_status

    def validate_data_types(self) -> bool:
        """
        Checks the data types of each column in the dataframe against the expected 
        data types defined in the schema.

        Returns:
        - bool: True if all column data types match the schema, False otherwise.
        """
        logger.info("Starting data type validation.")
        validation_status = True
        
        expected_data_types = {col: self.config.schema[col]['type'] for col in self.config.schema if col in self.df.columns}

        for column, dtype in expected_data_types.items():
            if not pd.api.types.is_dtype_equal(self.df[column].dtype, dtype):
                validation_status = False
                logger.warning(f"Data type mismatch for column '{column}': Expected {dtype} but got {self.df[column].dtype}")

        if validation_status:
            logger.info("All data types are as expected.")
        return validation_status


    def _write_status_to_file(self, message: str, overwrite: bool = False):
        """
        Writes the validation status message to a specified file.

        Args:
        - message (str): The message to write.
        - overwrite (bool, optional): If set to True, overwrites the file. If False, appends to the file.
        """
        logger.info("Writing validation status to file.")
        mode = 'w' if overwrite else 'a'
        try:
            with open(self.config.status_file, mode) as f:
                f.write(message + "\n")
        except Exception as e:
            logger.error(f"Error writing to status file: {e}")
            raise

    def run_all_validations(self) -> bool:
        """
        Executes all data validations and logs the overall status. 
        It encompasses both feature existence and data type checks.
        """
        logger.info("Running all data validations.")
        feature_validation_status = self.validate_all_features()
        data_type_validation_status = self.validate_data_types()

        overall_status = "Overall Validation Status: "
        if feature_validation_status and data_type_validation_status:
            overall_status += "All validations passed."
            logger.info(overall_status)
        else:
            overall_status += "Some validations failed. Check the log for details."
            logger.error(overall_status)
        
        self._write_status_to_file(overall_status)
        return feature_validation_status and data_type_validation_status
