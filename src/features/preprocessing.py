"""
Feature Preprocessing Module
Module for preprocessing features
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class FeaturePreprocessor:
    """
    Class for preprocessing features
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler()
        self.is_fitted = False
    
    def _get_scaler(self):
        """Select scaler based on specified method"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if self.scaling_method not in scalers:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        return scalers[self.scaling_method]
    
    def fit(self, X):
        """Fit scaler with training data"""
        self.scaler.fit(X)
        self.is_fitted = True
        print(f"✓ Fitted {self.scaling_method} scaler")
        return self
    
    def transform(self, X):
        """Transform data with fitted scaler"""
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        print(f"✓ Transformed data with {self.scaling_method} scaler")
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def get_stats(self):
        """Display scaler statistics"""
        if not self.is_fitted:
            return None
        
        if hasattr(self.scaler, 'mean_'):
            print("\n📊 Scaler Statistics:")
            print(f"  Mean: {self.scaler.mean_}")
            print(f"  Scale: {self.scaler.scale_}")
        elif hasattr(self.scaler, 'data_min_'):
            print("\n📊 Scaler Statistics:")
            print(f"  Min: {self.scaler.data_min_}")
            print(f"  Max: {self.scaler.data_max_}")


def preprocess_pipeline(X_train, X_test, method: str = 'standard'):
    """
    Pipeline for preprocessing data
    
    Args:
        X_train: training features
        X_test: test features
        method: scaling method
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, preprocessor)
    """
    preprocessor = FeaturePreprocessor(scaling_method=method)
    
    # Fit only with train data!
    X_train_scaled = preprocessor.fit_transform(X_train)
    
    # Transform test data with parameters from train
    X_test_scaled = preprocessor.transform(X_test)
    
    return X_train_scaled, X_test_scaled, preprocessor


if __name__ == "__main__":
    # Test module
    import sys
    sys.path.insert(0, '.')
    from src.data.load_data import load_sklearn_dataset, split_data
    
    # Load data
    X, y, features, targets = load_sklearn_dataset('iris')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\n" + "="*50)
    print("Testing StandardScaler")
    print("="*50)
    
    # Test StandardScaler
    X_train_scaled, X_test_scaled, preprocessor = preprocess_pipeline(
        X_train, X_test, method='standard'
    )
    
    preprocessor.get_stats()
    
    print(f"\n📈 Original X_train mean: {X_train.mean(axis=0)}")
    print(f"📉 Scaled X_train mean: {X_train_scaled.mean(axis=0)}")
    
    print("\n" + "="*50)
    print("Testing MinMaxScaler")
    print("="*50)
    
    # Test MinMaxScaler
    X_train_mm, X_test_mm, _ = preprocess_pipeline(
        X_train, X_test, method='minmax'
    )
    
    print(f"\n📈 MinMax X_train range: [{X_train_mm.min():.2f}, {X_train_mm.max():.2f}]")
