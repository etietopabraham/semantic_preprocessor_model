from dataclasses import dataclass
from pathlib import Path

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
