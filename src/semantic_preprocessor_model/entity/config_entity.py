from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion process.
    
    Attributes:
    - root_dir: Directory where data ingestion artifacts are stored.
    - local_data_file: Path to the local file where the data is already saved.
    """
    root_dir: Path  # Directory where data ingestion artifacts are stored
    local_data_file: Path  # Path to the local file where the data is already saved


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration class for data validation.
    
    This class captures the essential configurations required for data validation, 
    including directories for storing validation results, paths to data files, 
    and the expected data schema.
    
    Attributes:
    -----------
    root_dir : Path
        Directory for storing validation results and related artifacts.
        
    data_source_file : Path
        Path to the ingested or feature-engineered data file.
        
    status_file : Path
        File for logging the validation status.
        
    schema : Dict[str, Dict[str, str]]
        Dictionary containing initial schema configurations for data validation.
    """
    
    root_dir: Path  # Directory for storing validation results and related artifacts
    data_source_file: Path  # Path to the ingested or feature-engineered data file
    status_file: Path  # File for logging the validation status
    schema: Dict[str, Dict[str, str]]  # Dictionary containing initial schema configurations


