"""
Comprehensive Model Evaluation and Results Analysis
Compares baseline vs AI-enhanced ESG models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationFramework:
    """
    Comprehensive evaluation comparing baseline and AI-enhanced models
    """
    
    def __init__(self):
        self.baseline_results = {}
        self.enhanced_results = {}
        self.comparison_df = None
        
    def load_baseline_results(self, results_dict):
        """Load baseline model results"""
        self.baseline_results = results_dict
        print("✓ Baseline results loaded")
    
    def load_enhanced_results(self, results_dict):
        """Load AI-enhanced model results"""
        self.enhanced_results = results_dict
        print("✓ Enhanced results loaded")
    
    def create_comparison_table(self):
        """
        Create comprehensive comparison table
        
        Returns:
        --------
        pd.DataFrame : Comparison of all models
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        # Baseline models
        for model_name, metrics in self.baseline_results.items():
            comparison_data.append({
                'Model': f"Baseline - {model_name}",
                'Category': 'Traditional ML',
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Overfitting': metrics['train_r2'] - metrics['test_r2']
            })
        
        # AI-enhanced model
        if self.enhanced_results:
            metrics = self.enhanced_results['metrics']
            comparison_data.append({
                'Model': 'AI-Enhanced (Gradient Boosting)',
                'Category': 'Advanced AI',
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Overfitting': metrics['train_r2'] - metrics['test_r2']
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Test R²
        self.comparison_df = self.comparison_df.sort_values('Test R²', ascending=False)
        
        print("\n" + self.comparison_df.to_string(index=False))
        print("\n" + "="*60)
        
        return self.comparison_df
    
    def calculate_improvement_metrics(self):
        """Calculate improvement of AI-enhanced over best baseline"""
        if not self.enhanced_results or not self.baseline_results:
            print("Please load both baseline and enhanced results first")
            return None
        
        print("\n" + "="*60)
        print("AI-ENHANCED MODEL IMPROVEMENTS")
        print("="*60)
        
        # Find best baseline model
        best_baseline = max(self.baseline_results.items(), 
                          key=lambda x: x[1]['test_r2'])
        best_baseline_name = best_baseline[0]
        best_baseline_metrics = best_baseline[1]
        
        enhanced_metrics = self.enhanced_results['metrics']
        
        # Calculate improvements
        r2_improvement = ((enhanced_metrics['test_r2'] - best_baseline_metrics['test_r2']) 
                         / best_baseline_metrics['test_r2'] * 100)
        
        rmse_improvement = ((best_baseline_metrics['test_rmse'] - enhanced_metrics['test_rmse']) 
                          / best_baseline_metrics['test_rmse'] * 100)
        
        mae_improvement = ((best_baseline_metrics['test_mae'] - enhanced_metrics['test_mae']) 
                         / best_baseline_metrics['test_mae'] * 100)
        
        print(f"\nComparison vs Best Baseline ({best_baseline_name}):")
        print(f"  R² Improvement: {r2_improvement:+.2f}%")
        print(f"  RMSE Improvement: {rmse_improvement:+.2f}%")
        print(f"  MAE Improvement: {mae_improvement:+.2f}%")
        
        print(f"\nAbsolute Metrics:")
        print(f"  Best Baseline R²: {best_baseline_metrics['test_r2']:.4f}")
        print(f"  AI-Enhanced R²: {enhanced_metrics['test_r2']:.4f}")
        print(f"  Difference: {enhanced_metrics['test_r2'] - best_baseline_metrics['test_r2']:+.4f}")
        
        improvements = {
            'r2_improvement_pct': r2_improvement,
            'rmse_improvement_pct': rmse_improvement,
            'mae_improvement_pct': mae_improvement,
            'best_baseline_name': best_baseline_name,
            'best_baseline_r2': best_baseline_metrics['test_r2'],
            'enhanced_r2': enhanced_metrics['test_r2']
        }
        
        return improvements
    
    def plot_model_comparison(self):
        """Create visualization comparing all models"""
        if self.comparison_df is None:
            print("Please create comparison table first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. R² Score Comparison
        models = self.comparison_df['Model']
        test_r2 = self.comparison_df['Test R²']
        
        colors = ['#ff7f0e' if 'Baseline' in m else '#2ca02c' for m in models]
        
        axes[0, 0].barh(range(len(models)), test_r2, color=colors)
        axes[0, 0].set_yticks(range(len(models)))
        axes[0, 0].set_yticklabels(models, fontsize=9)
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model Performance: R² Score (Test Set)')
        axes[0, 0].axvline(x=0.8, color='red', linestyle='--', alpha=0.3, label='Target: 0.8')
        axes[0, 0].legend()
        axes[0, 0].invert_yaxis()
        
        # 2. RMSE Comparison
        test_rmse = self.comparison_df['Test RMSE']
        axes[0, 1].barh(range(len(models)), test_rmse, color=colors)
        axes[0, 1].set_yticks(range(len(models)))
        axes[0, 1].set_yticklabels(models, fontsize=9)
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Model Performance: RMSE (Test Set)')
        axes[0, 1].invert_yaxis()
        
        # 3. Overfitting Analysis
        overfitting = self.comparison_df['Overfitting']
        axes[1, 0].barh(range(len(models)), overfitting, color=colors)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels(models, fontsize=9)
        axes[1, 0].set_xlabel('Train R² - Test R² (Lower is Better)')
        axes[1, 0].set_title('Overfitting Analysis')
        axes[1, 0].axvline(x=0.1, color='red', linestyle='--', alpha=0.3, label='Acceptable: <0.1')
        axes[1, 0].legend()
        axes[1, 0].invert_yaxis()
        
        # 4. Train vs Test R²
        train_r2 = self.comparison_df['Train R²']
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_r2, width, label='Train R²', color='skyblue')
        axes[1, 1].bar(x + width/2, test_r2, width, label='Test R²', color='lightcoral')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Train vs Test Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comparison plots saved to 'results/model_comparison_comprehensive.png'")
        plt.close()
    
    def plot_residual_analysis(self):
        """Detailed residual analysis for AI-enhanced model"""
        if not self.enhanced_results:
            print("Please load enhanced results first")
            return
        
        y_test = self.enhanced_results['actuals']['test']
        y_pred = self.enhanced_results['predictions']['test']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Residual Plot
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residual Plot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Predicted vs Actual
        axes[0, 1].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual ESG Score')
        axes[0, 1].set_ylabel('Predicted ESG Score')
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/residual_analysis_enhanced_model.png', dpi=300, bbox_inches='tight')
        print("✓ Residual analysis plots saved to 'results/residual_analysis_enhanced_model.png'")
        plt.close()
    
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        print("\n" + "="*70)
        print(" MODEL PERFORMANCE SUMMARY ".center(70, "="))
        print("="*70)
        
        if self.comparison_df is None:
            self.create_comparison_table()
        
        improvements = self.calculate_improvement_metrics()
        
        # Statistical summary
        print("\n" + "-"*70)
        print("STATISTICAL ANALYSIS")
        print("-"*70)
        
        print(f"\nBaseline Models Performance:")
        baseline_models = self.comparison_df[self.comparison_df['Category'] == 'Traditional ML']
        print(f"  Average Test R²: {baseline_models['Test R²'].mean():.4f}")
        print(f"  Best Test R²: {baseline_models['Test R²'].max():.4f}")
        print(f"  Std Dev: {baseline_models['Test R²'].std():.4f}")
        
        print(f"\nAI-Enhanced Model Performance:")
        enhanced_row = self.comparison_df[self.comparison_df['Category'] == 'Advanced AI'].iloc[0]
        print(f"  Test R²: {enhanced_row['Test R²']:.4f}")
        print(f"  Test RMSE: {enhanced_row['Test RMSE']:.4f}")
        print(f"  Test MAE: {enhanced_row['Test MAE']:.4f}")
        print(f"  Overfitting Gap: {enhanced_row['Overfitting']:.4f}")
        
        # Key findings
        print("\n" + "-"*70)
        print("KEY FINDINGS")
        print("-"*70)
        
        findings = []
        
        if improvements['r2_improvement_pct'] > 5:
            findings.append(f"✓ AI-enhanced model shows {improvements['r2_improvement_pct']:.1f}% "
                          f"improvement in R² over best baseline")
        
        if enhanced_row['Overfitting'] < 0.1:
            findings.append("✓ Model demonstrates good generalization with minimal overfitting")
        
        if enhanced_row['Test R²'] > 0.85:
            findings.append("✓ Excellent predictive performance (R² > 0.85)")
        elif enhanced_row['Test R²'] > 0.75:
            findings.append("✓ Strong predictive performance (R² > 0.75)")
        
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        
        print("\n" + "="*70)
        
        return {
            'comparison_table': self.comparison_df,
            'improvements': improvements,
            'summary_stats': {
                'baseline_avg_r2': baseline_models['Test R²'].mean(),
                'enhanced_r2': enhanced_row['Test R²'],
                'performance_gain': improvements['r2_improvement_pct']
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluationFramework()
    
    # Load baseline results (from Step 4)
    baseline_results = {
        'Linear Regression': {
            'train_r2': 0.7234,
            'test_r2': 0.7012,
            'train_rmse': 8.45,
            'test_rmse': 8.89,
            'test_mae': 6.78
        },
        'Random Forest': {
            'train_r2': 0.8876,
            'test_r2': 0.8234,
            'train_rmse': 5.23,
            'test_rmse': 6.78,
            'test_mae': 5.12
        },
        'Decision Tree': {
            'train_r2': 0.8534,
            'test_r2': 0.7856,
            'train_rmse': 6.12,
            'test_rmse': 7.45,
            'test_mae': 5.89
        }
    }
    
    # Load AI-enhanced results (from Step 5)
    enhanced_results = {
        'model': None,  # Model object
        'metrics': {
            'train_r2': 0.9123,
            'test_r2': 0.8845,
            'train_rmse': 4.56,
            'test_rmse': 5.34,
            'test_mae': 4.12
        },
        'predictions': {
            'train': np.random.randn(100),  # Placeholder
            'test': np.random.randn(20)
        },
        'actuals': {
            'train': np.random.randn(100),
            'test': np.random.randn(20)
        }
    }
    
    evaluator.load_baseline_results(baseline_results)
    evaluator.load_enhanced_results(enhanced_results)
    
    # Generate comprehensive evaluation
    summary = evaluator.generate_performance_summary()
    
    # Create visualizations
    evaluator.plot_model_comparison()
    evaluator.plot_residual_analysis()
    
    # Save comparison table
    evaluator.comparison_df.to_csv('results/model_comparison_table.csv', index=False)
    print("\n✓ Comparison table saved to 'results/model_comparison_table.csv'")
    
    print("\n" + "="*70)
    print("✓ MODEL EVALUATION COMPLETE")
    print("="*70)
