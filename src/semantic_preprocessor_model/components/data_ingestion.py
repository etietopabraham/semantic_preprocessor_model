import os
import shutil
from src.semantic_preprocessor_model import logger
from src.semantic_preprocessor_model.utils.common import get_size
from semantic_preprocessor_model.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    DataIngestion handles the process of transferring data from a local directory 
    to the project's official artifact directories.

    The class currently assumes that the data is already present locally, 
    and focuses on transferring this data to the specified directory.

    Attributes:
    - config (DataIngestionConfig): Configuration settings for data ingestion.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion component.

        Args:
        - config (DataIngestionConfig): Configuration settings for data ingestion.
        """
        self.config = config

    def download_data(self):
        """ 
        Placeholder for downloading data functionality. 
        Currently, data is assumed to be locally available.
        """
        pass

    def extract_zip_file(self):
        """
        Placeholder for extracting zip files. 
        If the data comes as a zip file, this method can be used to extract it.
        """
        pass

    def transfer_data(self) -> None:
        """
        Transfer the data from the local directory to the project's artifact directory.

        This method ensures that the artifact directory exists, and then transfers 
        the data file to this directory.

        Raises:
        - FileNotFoundError: If the local data file does not exist.
        """
        root_dir = Path(self.config.root_dir)
        local_data_path = Path(self.config.local_data_file)
        
        # Check if the local data file exists
        if not local_data_path.exists():
            logger.error(f"Local data file not found at {local_data_path}.")
            raise FileNotFoundError(f"No file found at {local_data_path}")

        # Get the file size using the utility function
        file_size = get_size(local_data_path)

        # Ensure the transfer directory exists
        os.makedirs(root_dir, exist_ok=True)

        # Transfer the file
        shutil.copy2(local_data_path, root_dir)
        logger.info(f"Data transferred from {local_data_path} to {root_dir}. File size: {file_size}.")