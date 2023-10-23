import pandas as pd
from src.semantic_preprocessor_model import logger
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
import nltk
import pickle

from src.semantic_preprocessor_model.config.configuration import DataTransformationConfig



class DataTransformation:
    """
    Class for transforming ingested datasets through:
    - Text preprocessing
    - Missing value handling
    - TF-IDF vectorization for 'work_name' text feature
    - Data filtering
    - Data splitting for training and validation
    - Storing transformed datasets
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

        # Initialize the TF-IDF vectorizer
        self.vectorizer_work_name = TfidfVectorizer(max_features=5000)


    @classmethod
    def from_dataframe(cls, config: DataTransformationConfig, df: pd.DataFrame):
        """
        Alternate constructor that initializes the DataTransformation component using a DataFrame.
        
        Args:
        - config (DataTransformationConfig): Configuration settings for data transformation.
        - df (pd.DataFrame): The DataFrame containing the dataset.
        
        Returns:
        - DataTransformation: An instance of the DataTransformation class.
        """
        instance = cls(config)
        instance.df = df
        return instance
    
    @classmethod
    def load_vectorizer(cls, path: str):
        """
        Load a trained vectorizer from a file.
        
        Args:
        - path (str): Path to load the trained vectorizer from.

        Returns:
        - Trained vectorizer.
        """
        with open(path, "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer
    
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from the source file."""
        try:
            return pd.read_csv(self.config.data_source_file)
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_source_file}")
            raise

    def _download_nltk_resources(self):
        """Ensure NLTK resources are available."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def _initialize_stop_words(self):
        """Initialize Russian stop words."""
        self.stop_words = set(stopwords.words('russian'))

    def _handle_missing_values(self):
        """Handle NaN values in 'upper_works' column."""
        self.df['upper_works'].fillna('Unknown', inplace=True)

    def preprocess_text(self, text: str) -> str:
        """
        Tokenize, lowercase, and filter out stop words.
        
        Args:
        - text (str): Raw text.

        Returns:
        - str: Processed text.
        """
        tokens = word_tokenize(text.lower(), language='russian')
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return ' '.join(tokens)

    def apply_text_preprocessing(self):
        """Process 'work_name' column texts."""
        self.df['processed_work_name'] = self.df['work_name'].apply(self.preprocess_text)

    def vectorize_text_features(self) -> csr_matrix:
        """
        Vectorize 'processed_work_name' using TF-IDF.
        
        Returns:
        - csr_matrix: Vectorized features.
        """
        tfidf_work_name = self.vectorizer_work_name.fit_transform(self.df['processed_work_name'])
        return tfidf_work_name.tocsr()

    def filter_data(self, combined_tfidf_features_csr) -> (pd.DataFrame, csr_matrix):
        """
        Filter out singleton classes and rows with missing 'generalized_work_class' values.
        
        Args:
        - combined_tfidf_features_csr (csr_matrix): Vectorized features.

        Returns:
        - pd.DataFrame, csr_matrix: Filtered data and features.
        """
        trainable_data = self.df[~self.df['generalized_work_class'].isnull()]
        class_counts = trainable_data['generalized_work_class'].value_counts()
        singleton_classes = class_counts[class_counts == 1]
        filtered_data = trainable_data[~trainable_data['generalized_work_class'].isin(singleton_classes.index)]
        return filtered_data, combined_tfidf_features_csr[filtered_data.index, :]

    def split_data(self, X, y) -> tuple:
        """
        Stratified split for training and test sets.
        
        Args:
        - X (csr_matrix): Features.
        - y (pd.Series): Labels.

        Returns:
        - tuple: Train and test datasets.
        """
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def _save_datasets(self, X_train, y_train, X_val, y_val, train_features_filename: str, train_labels_filename: str, val_features_filename: str, val_labels_filename: str):
        """
        Save datasets to paths in sparse format.
        
        Args:
        - X_train (csr_matrix): Training features.
        - y_train (pd.Series): Training labels.
        - X_val (csr_matrix): Validation features.
        - y_val (pd.Series): Validation labels.
        - train_features_filename (str): File name for training features.
        - train_labels_filename (str): File name for training labels.
        - val_features_filename (str): File name for validation features.
        - val_labels_filename (str): File name for validation labels.
        """
        train_features_output_path = self.config.root_dir / train_features_filename
        train_labels_output_path = self.config.root_dir / train_labels_filename
        val_features_output_path = self.config.root_dir / val_features_filename
        val_labels_output_path = self.config.root_dir / val_labels_filename
        
        try:
            # Store training and validation features as csr matrices
            save_npz(train_features_output_path, X_train)
            save_npz(val_features_output_path, X_val)
            logger.info(f"Stored training and validation features in NPZ format.")

            # Store training and validation labels as CSV files
            y_train.to_csv(train_labels_output_path, index=False)
            y_val.to_csv(val_labels_output_path, index=False)
            logger.info(f"Stored training and validation labels in CSV format.")

        except Exception as e:
            logger.error(f"Error while saving datasets: {e}")


    def save_vectorizer(self, path: str):
        """
        Save the trained vectorizer to a file.
        
        Args:
        - path (str): Path to save the trained vectorizer.
        """
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer_work_name, f)
        logger.info(f"Vectorizer saved to {path}")


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

        # Save the trained vectorizer
        self.save_vectorizer("/Users/macbookpro/Documents/semantic_preprocessor_model/semantic_preprocessor_model/artifacts/vectorizersvectorizer.pkl")

        logger.info("Filtering combined features")
        filtered_data, filtered_features = self.filter_data(combined_tfidf_features_csr)

        logger.info("Splitting data")
        X_train, X_val, y_train, y_val = self.split_data(filtered_features, filtered_data['generalized_work_class'])

        logger.info("Saving to artifacts")
        self._save_datasets(X_train, y_train, X_val, y_val, train_features_filename, train_labels_filename, val_features_filename, val_labels_filename)
        
        return X_train, X_val, y_train, y_val
    
    def transform_test_data(self, test_df: pd.DataFrame) -> tuple:
        """
        Transform the test dataset by applying text preprocessing, vectorization, and other necessary steps.

        Args:
        - test_df (pd.DataFrame): The test dataset.

        Returns:
        - tuple: Transformed test features and labels.
        """
        
        # Check if the vectorizer is trained
        if not hasattr(self.vectorizer_work_name, 'vocabulary_'):
            raise RuntimeError("Vectorizer is not trained. Ensure 'transform' method is called before 'transform_test_data'.")
        
        # Drop records with missing 'generalized_work_class' and 'global_work_class'
        test_df.dropna(subset=['generalized_work_class', 'global_work_class'], inplace=True)
        
        # 1. Apply text preprocessing
        logger.info("Preprocessing work_name")
        test_df['processed_work_name'] = test_df['work_name'].apply(self.preprocess_text)
        
        # 2. Use the trained vectorizer to transform test data
        logger.info("Vectorizing processed work name with same vectorizer settings")
        tfidf_work_name = self.vectorizer_work_name.transform(test_df['processed_work_name'])
        
        X_test = tfidf_work_name.tocsr()
        y_test = test_df['generalized_work_class']
        
        return X_test, y_test



