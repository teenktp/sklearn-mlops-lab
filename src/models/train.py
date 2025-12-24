"""
Model Training Module
Module for training ML models - Random Forest Version
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime


class ModelTrainer:
    """
    Class for training and evaluating models
    """
    
    def __init__(self, model_name: str = 'random_forest', **kwargs):
        """
        Initialize trainer
        
        Args:
            model_name: Model name
            **kwargs: Hyperparameters for model
        """
        self.model_name = model_name
        self.hyperparameters = kwargs
        self.model = self._get_model()
        self.is_trained = False
        self.training_history = {}
    
    def _get_model(self):
        """Create model instance"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        # Merge default with user params
        params = {**default_params, **self.hyperparameters}
        
        if self.model_name == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def train(self, X_train, y_train):
        """Train model"""
        print(f"�� Training {self.model_name}...")
        print(f"   Hyperparameters: {self.hyperparameters}")
        
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        end_time = datetime.now()
        
        self.is_trained = True
        self.training_history['training_time'] = (end_time - start_time).total_seconds()
        self.training_history['n_samples'] = X_train.shape[0]
        self.training_history['n_features'] = X_train.shape[1]
        self.training_history['hyperparameters'] = self.hyperparameters
        
        # Train accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        self.training_history['train_accuracy'] = train_acc
        
        print(f"✓ Training completed in {self.training_history['training_time']:.2f}s")
        print(f"  Train Accuracy: {train_acc:.4f}")
        
        return self
    
    def evaluate(self, X_test, y_test, target_names=None):
        """Evaluate model"""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        
        print(f"\n📊 Evaluating {self.model_name}...")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, y_pred)
        self.training_history['test_accuracy'] = test_acc
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n🎯 Feature Importances:")
            importances = self.model.feature_importances_
            for i, imp in enumerate(importances):
                print(f"  Feature {i}: {imp:.4f}")
        
        return y_pred, test_acc
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        
        if filepath is None:
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'models/{self.model_name}_{timestamp}.joblib'
        
        joblib.dump({
            'model': self.model,
            'model_name': self.model_name,
            'training_history': self.training_history
        }, filepath)
        
        print(f"✓ Model saved to: {filepath}")
        return filepath
    
    def get_summary(self):
        """Display training summary"""
        print("\n" + "="*50)
        print(f"🌲 Training Summary: {self.model_name}")
        print("="*50)
        for key, value in self.training_history.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    - {k}: {v}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # Test module
    import sys
    sys.path.insert(0, '.')
    from src.data.load_data import load_sklearn_dataset, split_data
    
    # Load data
    X, y, features, targets = load_sklearn_dataset('iris')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Test different hyperparameters
    experiments = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': None}
    ]
    
    results = []
    
    for exp in experiments:
        print("\n" + "="*60)
        trainer = ModelTrainer('random_forest', **exp)
        trainer.train(X_train, y_train)
        y_pred, accuracy = trainer.evaluate(X_test, y_test, target_names=targets)
        results.append({
            'params': exp,
            'test_accuracy': accuracy
        })
    
    # Summary of all experiments
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)
    for i, result in enumerate(results):
        print(f"\nExperiment {i+1}:")
        print(f"  Params: {result['params']}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
