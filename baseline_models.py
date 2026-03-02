"""
Baseline Machine Learning Models for ESG Risk Assessment
Establishes benchmark performance before AI-enhanced methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """
    Implements and evaluates baseline ML models for ESG prediction
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, df, target_variable='weighted_esg_score', task='regression'):
        """
        Prepare features and target for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        target_variable : str
            Target variable to predict
        task : str
            'regression' or 'classification'
        
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("DATA PREPARATION FOR MODELING")
        print("="*60)
        
        # Select features (exclude identifiers and target)
        exclude_cols = ['ticker', 'company_name', 'sector', 'industry', 'country', 
                       target_variable, 'esg_category']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_variable]
        
        # For classification, convert to binary or multiclass
        if task == 'classification':
            if 'esg_category' in df.columns:
                y = df['esg_category']
            else:
                # Create binary classification: High risk (0) vs Low risk (1)
                y = (y > y.median()).astype(int)
                print("  Created binary target: High Risk (0) vs Low Risk (1)")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"  Features selected: {len(feature_cols)}")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Testing samples: {len(self.X_test)}")
        print(f"  Target variable: {target_variable}")
        print(f"  Task type: {task}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_regression_models(self):
        self.results = {}
        self.models = {}

        """
        Train baseline regression models
        
        Returns:
        --------
        dict : Trained models and their metrics
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE REGRESSION MODELS")
        print("="*60)
        
        models_config = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Metrics
            metrics = {
                'model': model,
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_r2': r2_score(self.y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'predictions': y_test_pred
            }
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"  ✓ Training R²: {metrics['train_r2']:.4f}")
            print(f"  ✓ Testing R²: {metrics['test_r2']:.4f}")
            print(f"  ✓ Testing RMSE: {metrics['test_rmse']:.4f}")
            print(f"  ✓ Testing MAE: {metrics['test_mae']:.4f}")
        
        return self.results
    
    def train_classification_models(self):
        self.results = {}  
        self.models={}

        """
        Train baseline classification models
        
        Returns:
        --------
        dict : Trained models and their metrics
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE CLASSIFICATION MODELS")
        print("="*60)
        
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree Classifier': DecisionTreeClassifier(max_depth=10, random_state=42)
        }
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Metrics
            metrics = {
                'model': model,
                'train_accuracy': accuracy_score(self.y_train, y_train_pred),
                'test_accuracy': accuracy_score(self.y_test, y_test_pred),
                'train_precision': precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
                'train_f1': f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
                'test_f1': f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(self.y_test, y_test_pred),
                'predictions': y_test_pred
            }
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"  ✓ Training Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  ✓ Testing Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  ✓ Testing Precision: {metrics['test_precision']:.4f}")
            print(f"  ✓ Testing Recall: {metrics['test_recall']:.4f}")
            print(f"  ✓ Testing F1-Score: {metrics['test_f1']:.4f}")
        
        return self.results
    
    def generate_results_table(self, task='regression'):
        """
        Generate formatted results table
        
        Parameters:
        -----------
        task : str
            'regression' or 'classification'
        
        Returns:
        --------
        pd.DataFrame : Results summary table
        """
        if task == 'regression':
            results_data = []
            for model_name, metrics in self.results.items():
                results_data.append({
                    'Model': model_name,
                    'Train R²': f"{metrics['train_r2']:.4f}",
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    'Train RMSE': f"{metrics['train_rmse']:.4f}",
                    'Test RMSE': f"{metrics['test_rmse']:.4f}",
                    'Test MAE': f"{metrics['test_mae']:.4f}"
                })
        else:
            results_data = []
            for model_name, metrics in self.results.items():
                results_data.append({
                    'Model': model_name,
                    'Train Acc': f"{metrics['train_accuracy']:.4f}",
                    'Test Acc': f"{metrics['test_accuracy']:.4f}",
                    'Precision': f"{metrics['test_precision']:.4f}",
                    'Recall': f"{metrics['test_recall']:.4f}",
                    'F1-Score': f"{metrics['test_f1']:.4f}"
                })
        
        return pd.DataFrame(results_data)
    
    def plot_regression_results(self, model_name='Random Forest'):
        """
        Plot residuals and predictions for regression model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return
        
        predictions = self.results[model_name]['predictions']
        residuals = self.y_test - predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual plot
        axes[0].scatter(predictions, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residual Plot - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Predicted vs Actual
        axes[1].scatter(self.y_test, predictions, alpha=0.6)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2)
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title(f'Predicted vs Actual - {model_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/baseline_{model_name.lower().replace(" ", "_")}_regression.png', dpi=300, bbox_inches='tight')
        print(f"✓ Regression plots saved")
        plt.close()
    
    def plot_confusion_matrix(self, model_name='Random Forest Classifier'):
        """
        Plot confusion matrix for classification model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'results/baseline_{model_name.lower().replace(" ", "_")}_confusion.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Load processed data
    try:
        processed_data = pd.read_csv('data/processed/esg_processed_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run preprocessing first.")
        exit()
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # REGRESSION TASK
    print("\n" + "="*70)
    print(" BASELINE MODELS: REGRESSION TASK ".center(70, "="))
    print("="*70)
    
    baseline.prepare_data(processed_data, target_variable='weighted_esg_score', task='regression')
    baseline.train_regression_models()
    
    # Display results
    results_table = baseline.generate_results_table(task='regression')
    print("\n" + "="*60)
    print("REGRESSION RESULTS SUMMARY")
    print("="*60)
    print(results_table.to_string(index=False))
    
    # Plot results
    baseline.plot_regression_results('Random Forest')
    
    # CLASSIFICATION TASK
    print("\n" + "="*70)
    print(" BASELINE MODELS: CLASSIFICATION TASK ".center(70, "="))
    print("="*70)
    
    baseline.prepare_data(processed_data, target_variable='weighted_esg_score', task='classification')
    baseline.train_classification_models()
    
    # Display results
    results_table_clf = baseline.generate_results_table(task='classification')
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*60)
    print(results_table_clf.to_string(index=False))
    
    # Plot confusion matrix
    baseline.plot_confusion_matrix('Random Forest Classifier')
    
    print("\n" + "="*70)
    print("✓ BASELINE MODELS COMPLETE")
    print("="*70)
