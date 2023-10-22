from src.semantic_preprocessor_model import logger
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
import nltk

from src.semantic_preprocessor_model.config.configuration import DataTransformationConfig

class DataTransformation:
    """
    Class responsible for transforming the ingested dataset through various processes:
    - Text preprocessing
    - Handling missing values
    - TF-IDF vectorization of text features
    - Data filtering
    - Dataset splitting
    - Saving the transformed datasets
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component, loads data, and sets up prerequisites.
        
        Args:
        - config (DataTransformationConfig): Configuration settings for data transformation.
        """
        self.config = config
        self.df = self._load_data()
        self._download_nltk_resources()
        self._initialize_stop_words()
        self._handle_missing_values()

    def _load_data(self) -> pd.DataFrame:
        """
        Load data from the specified source file.

        Returns:
        - pd.DataFrame: Loaded data.
        """
        try:
            return pd.read_csv(self.config.data_source_file)
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_source_file}")
            raise

    def _download_nltk_resources(self):
        """Download necessary NLTK resources if they aren't present."""
        if not nltk.data.find('tokenizers/punkt'):
            nltk.download('punkt')
        if not nltk.data.find('corpora/stopwords'):
            nltk.download('stopwords')

    def _initialize_stop_words(self):
        """Initialize a set of Russian stop words."""
        self.stop_words = set(stopwords.words('russian'))

    def _handle_missing_values(self):
        """Handle missing values by replacing NaN in 'upper_works' column with 'Unknown'."""
        self.df['upper_works'].fillna('Unknown', inplace=True)

    def preprocess_text(self, text: str) -> str:
        """
        Tokenize, convert to lowercase, and filter out stop words from the text.
        
        Args:
        - text (str): Input text.

        Returns:
        - str: Processed text.
        """
        tokens = word_tokenize(text.lower(), language='russian')
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return ' '.join(tokens)

    def apply_text_preprocessing(self):
        """Apply text preprocessing to 'work_name' and 'upper_works' columns."""
        self.df['processed_work_name'] = self.df['work_name'].apply(self.preprocess_text)
        self.df['processed_upper_works'] = self.df['upper_works'].apply(self.preprocess_text)

    def vectorize_text_features(self) -> csr_matrix:
        """
        Vectorize text features using TF-IDF and combine them.

        Returns:
        - csr_matrix: Combined TF-IDF features.
        """
        vectorizer_work_name = TfidfVectorizer(max_features=5000)
        tfidf_work_name = vectorizer_work_name.fit_transform(self.df['processed_work_name'])

        vectorizer_upper_works = TfidfVectorizer(max_features=5000)
        tfidf_upper_works = vectorizer_upper_works.fit_transform(self.df['processed_upper_works'])

        return hstack([tfidf_work_name, tfidf_upper_works]).tocsr()

    def filter_data(self, combined_tfidf_features_csr) -> (pd.DataFrame, csr_matrix):
        """
        Filter out singleton classes and rows with missing 'generalized_work_class' values.

        Args:
        - combined_tfidf_features_csr (csr_matrix): Combined TF-IDF features.

        Returns:
        - pd.DataFrame, csr_matrix: Filtered data and corresponding TF-IDF features.
        """
        trainable_data = self.df[~self.df['generalized_work_class'].isnull()]
        class_counts = trainable_data['generalized_work_class'].value_counts()
        singleton_classes = class_counts[class_counts == 1]
        filtered_data = trainable_data[~trainable_data['generalized_work_class'].isin(singleton_classes.index)]
        return filtered_data, combined_tfidf_features_csr[filtered_data.index, :]

    def split_data(self, X, y) -> tuple:
        """
        Split data into training and test sets with stratification.

        Args:
        - X (csr_matrix): Features.
        - y (pd.Series): Labels.

        Returns:
        - tuple: Training and test datasets.
        """
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def _save_datasets(self, X_train, y_train, X_val, y_val, train_features_filename: str, train_labels_filename: str, val_features_filename: str, val_labels_filename: str):
        """
        Save transformed datasets to specified paths in sparse format.

        Args:
        - X_train (csr_matrix): Training features.
        - y_train (pd.Series): Training labels.
        - X_val (csr_matrix): Validation features.
        - y_val (pd.Series): Validation labels.
        - train_features_filename (str): Name of the file for saving training features.
        - train_labels_filename (str): Name of the file for saving training labels.
        - val_features_filename (str): Name of the file for saving validation features.
        - val_labels_filename (str): Name of the file for saving validation labels.
        """
        train_features_output_path = self.config.root_dir / train_features_filename
        train_labels_output_path = self.config.root_dir / train_labels_filename
        val_features_output_path = self.config.root_dir / val_features_filename
        val_labels_output_path = self.config.root_dir / val_labels_filename
        
        try:
            # Save the training and validation features as csr matrices
            save_npz(train_features_output_path, X_train)
            save_npz(val_features_output_path, X_val)
            logger.info(f"Training and validation features saved in NPZ format.")

            # Save the training and validation labels as CSV files
            y_train.to_csv(train_labels_output_path, index=False)
            y_val.to_csv(val_labels_output_path, index=False)
            logger.info(f"Training and validation labels saved in CSV format.")

        except Exception as e:
            logger.error(f"Error while saving datasets: {e}")


    def transform(self, 
                train_features_filename: str = "train_features.npz", 
                train_labels_filename: str = "train_labels.csv", 
                val_features_filename: str = "val_features.npz", 
                val_labels_filename: str = "val_labels.csv") -> tuple:
        """
        Execute entire data transformation pipeline.

        Args:
        - train_features_filename (str): Name for saving training features (default: "train_features.npz").
        - train_labels_filename (str): Name for saving training labels (default: "train_labels.csv").
        - val_features_filename (str): Name for saving validation features (default: "val_features.npz").
        - val_labels_filename (str): Name for saving validation labels (default: "val_labels.csv").

        Returns:
        - tuple: Transformed training and validation datasets.
        """
        logger.info("Applying text processing")
        self.apply_text_preprocessing()

        logger.info("Vectorizing text features")
        combined_tfidf_features_csr = self.vectorize_text_features()

        logger.info("Filtering combined features")
        filtered_data, filtered_features = self.filter_data(combined_tfidf_features_csr)

        logger.info("Splitting data")
        X_train, X_val, y_train, y_val = self.split_data(filtered_features, filtered_data['generalized_work_class'])

        logger.info("Saving to artifacts")
        self._save_datasets(X_train, y_train, X_val, y_val, train_features_filename, train_labels_filename, val_features_filename, val_labels_filename)
        
        return X_train, X_val, y_train, y_val