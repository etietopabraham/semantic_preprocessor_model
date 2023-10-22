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


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the data transformation process.
    
    This configuration class captures the necessary paths and directories 
    required for the transformation of data post-ingestion and pre-model training.
    
    Attributes:
    - root_dir: Directory where data transformation results and artifacts are stored.
    - data_source_file: Path to the file where the ingested data is stored that needs to be transformed.
    """
    
    root_dir: Path  # Directory for storing transformation results and related artifacts
    data_source_file: Path  # Path to the ingested data file for transformation
    data_validation: Path # Path to the validated output file


@dataclass(frozen=True)
class ModelTrainingConfig:
    """
    Configuration class for model training using the MLPClassifier neural network.

    Attributes:
    - root_dir: Directory for storing model training results and related artifacts.
    - train_features_path: Path to the features of the training dataset.
    - train_labels_path: Path to the labels of the training dataset.
    - val_features_path: Path to the features of the validation dataset.
    - val_labels_path: Path to the labels of the validation dataset.
    - model_name: Name or path under which the trained model will be saved.
    - hidden_layer_sizes: Number of neurons in each hidden layer.
    - max_iter: Maximum number of iterations for the solver to converge.
    - random_state: Seed for reproducibility.
    """
    
    root_dir: Path  # Directory for storing model training results and related artifacts
    train_features_path: Path  # Path to the features of the training dataset
    train_labels_path: Path  # Path to the labels of the training dataset
    val_features_path: Path  # Path to the features of the validation dataset
    val_labels_path: Path  # Path to the labels of the validation dataset
    model_name: str  # Name or path where the trained model should be saved
    
    hidden_layer_sizes: tuple  # Number of neurons in each hidden layer
    max_iter: int  # Maximum number of iterations for the solver to converge
    random_state: int  # Seed for reproducibility
