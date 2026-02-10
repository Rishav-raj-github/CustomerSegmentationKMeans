"""Data preprocessing and feature engineering module."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and feature engineering."""

    def __init__(self, scaling_method='standard'):
        """Initialize the preprocessor.
        
        Args:
            scaling_method (str): 'standard' or 'minmax' scaling method
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names = None
        self.imputer = SimpleImputer(strategy='mean')
        logger.info(f"DataPreprocessor initialized with {scaling_method} scaling")

    def load_data(self, filepath):
        """Load data from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df):
        """Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        return df

    def scale_features(self, X):
        """Scale features using specified method.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        return self.scaler.fit_transform(X)

    def fit_transform(self, df):
        """Fit and transform data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            np.ndarray: Preprocessed feature matrix
        """
        # Store feature names for later reference
        numeric_df = df.select_dtypes(include=[np.number])
        self.feature_names = numeric_df.columns.tolist()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Extract numeric features
        X = numeric_df.values
        
        # Scale features
        X_scaled = self.scale_features(X)
        
        logger.info(f"Data preprocessing complete: {X_scaled.shape}")
        return X_scaled
