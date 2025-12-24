"""
Results Logging Module
Module for logging experiment results
"""

import json
import csv
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Class for logging experiment results
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize logger
        
        Args:
            results_dir: Folder for storing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file for storing results
        self.csv_file = self.results_dir / 'experiments.csv'
        self._init_csv()
    
    def _init_csv(self):
        """Create CSV header if not exists"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'experiment_name',
                    'model_name',
                    'dataset',
                    'train_accuracy',
                    'test_accuracy',
                    'hyperparameters',
                    'notes'
                ])
            print(f"✓ Created results CSV: {self.csv_file}")
    
    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        dataset: str,
        train_accuracy: float,
        test_accuracy: float,
        hyperparameters: dict = None,
        notes: str = ''
    ):
        """
        Log experiment results
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                experiment_name,
                model_name,
                dataset,
                f'{train_accuracy:.4f}',
                f'{test_accuracy:.4f}',
                json.dumps(hyperparameters) if hyperparameters else '',
                notes
            ])
        
        print(f"✓ Logged experiment: {experiment_name}")
    
    def get_all_results(self):
        """Read all results"""
        results = []
        
        if self.csv_file.exists():
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                results = list(reader)
        
        return results
    
    def get_best_experiment(self, metric: str = 'test_accuracy'):
        """Find best experiment"""
        results = self.get_all_results()
        
        if not results:
            return None
        
        best = max(results, key=lambda x: float(x[metric]))
        return best
    
    def print_summary(self):
        """Display results summary"""
        results = self.get_all_results()
        
        if not results:
            print("📭 No experiments logged yet")
            return
        
        print("\n" + "="*70)
        print("📊 Experiment Results Summary")
        print("="*70)
        print(f"{'Experiment':<20} {'Model':<15} {'Train Acc':<12} {'Test Acc':<12}")
        print("-"*70)
        
        for r in results:
            print(f"{r['experiment_name']:<20} {r['model_name']:<15} "
                  f"{r['train_accuracy']:<12} {r['test_accuracy']:<12}")
        
        # Best experiment
        best = self.get_best_experiment()
        if best:
            print("\n🏆 Best Experiment:")
            print(f"   {best['experiment_name']} - Test Accuracy: {best['test_accuracy']}")


if __name__ == "__main__":
    # Test logger
    logger = ExperimentLogger()
    
    # Log sample experiments
    logger.log_experiment(
        experiment_name='baseline-lr',
        model_name='logistic_regression',
        dataset='iris',
        train_accuracy=0.975,
        test_accuracy=0.967,
        hyperparameters={'max_iter': 200},
        notes='Baseline experiment'
    )
    
    logger.log_experiment(
        experiment_name='rf-100trees',
        model_name='random_forest',
        dataset='iris',
        train_accuracy=1.0,
        test_accuracy=0.933,
        hyperparameters={'n_estimators': 100, 'max_depth': 5},
        notes='Random Forest with 100 trees'
    )
    
    logger.log_experiment(
        experiment_name='rf-200trees',
        model_name='random_forest',
        dataset='iris',
        train_accuracy=1.0,
        test_accuracy=0.967,
        hyperparameters={'n_estimators': 200, 'max_depth': None},
        notes='Random Forest with 200 trees'
    )
    
    # Show summary
    logger.print_summary()
