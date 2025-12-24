#!/usr/bin/env python3
"""
Main Experiment Pipeline
Run complete experiment from config
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from src.data.load_data import load_sklearn_dataset, split_data
from src.features.preprocessing import preprocess_pipeline
from src.utils.config import load_config, get_model_config


def run_experiment(config_path: str = 'configs/experiment_config.yaml'):
    """
    Run experiment based on config
    """
    print("="*60)
    print("🚀 Starting ML Experiment Pipeline")
    print("="*60)
    
    # 1. Load config
    config = load_config(config_path)
    
    # 2. Load data
    print("\n📥 Loading Data...")
    dataset_config = config['dataset']
    X, y, features, targets = load_sklearn_dataset(dataset_config['name'])
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=dataset_config['test_size'],
        random_state=dataset_config['random_state']
    )
    
    # 3. Preprocess
    print("\n⚙️ Preprocessing...")
    prep_config = config['preprocessing']
    X_train_scaled, X_test_scaled, preprocessor = preprocess_pipeline(
        X_train, X_test,
        method=prep_config['scaling_method']
    )
    
    # 4. Summary
    print("\n" + "="*60)
    print("✅ Pipeline Summary")
    print("="*60)
    print(f"  Dataset: {dataset_config['name']}")
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Scaling: {prep_config['scaling_method']}")
    print(f"  Classes: {list(targets)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, targets


def main():
    parser = argparse.ArgumentParser(description='Run ML Experiment')
    parser.add_argument(
        '--config', '-c',
        default='configs/experiment_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.config)
        print("\n🎉 Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
