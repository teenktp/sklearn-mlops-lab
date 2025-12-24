"""
Data Loading Module
Module for loading and managing data
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split


def load_sklearn_dataset(name: str = 'iris') -> tuple:
    """
    Load dataset from sklearn
    
    Args:
        name: Dataset name ('iris', 'wine', 'breast_cancer')
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(datasets.keys())}")
    
    data = datasets[name]()
    
    print(f"✓ Loaded {name} dataset")
    print(f"  Samples: {data.data.shape[0]}")
    print(f"  Features: {data.data.shape[1]}")
    print(f"  Classes: {len(data.target_names)}")
    
    return data.data, data.target, data.feature_names, data.target_names


def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into train and test sets
    
    Args:
        X: features
        y: targets
        test_size: Proportion of test set
        random_state: Seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split completed")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def create_dataframe(X, y, feature_names) -> pd.DataFrame:
    """
    Create DataFrame from numpy arrays
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


if __name__ == "__main__":
    # Test module
    X, y, features, targets = load_sklearn_dataset('iris')
    X_train, X_test, y_train, y_test = split_data(X, y)
    df = create_dataframe(X, y, features)
    print(f"\n📊 DataFrame shape: {df.shape}")
    print(df.head())
