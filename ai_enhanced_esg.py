"""
AI-Enhanced ESG Risk Assessment Model
Core contribution: Integrates domain knowledge with advanced ML techniques
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AIEnhancedESGModel:
    """
    Advanced ESG risk assessment combining ML, clustering, and rule-based logic
    """
    
    def __init__(self):
        self.esg_model = None
        self.clustering_model = None
        self.anomaly_detector = None
        self.feature_importance = None
        self.risk_flags = {}
        
    def calculate_weighted_esg_score(self, df, weights={'E': 0.40, 'S': 0.30, 'G': 0.30}):
        """
        Calculate weighted ESG composite score with domain-specific weights
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with E, S, G scores
        weights : dict
            Pillar weights (must sum to 1.0)
        
        Returns:
        --------
        pd.DataFrame : DataFrame with composite ESG score
        """
        print("\n" + "="*60)
        print("CALCULATING WEIGHTED ESG COMPOSITE SCORE")
        print("="*60)
        
        df_copy = df.copy()
        
        # Ensure weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
        
        # Calculate weighted score
        df_copy['composite_esg_score'] = (
            weights['E'] * df_copy['environment_score'] +
            weights['S'] * df_copy['social_score'] +
            weights['G'] * df_copy['governance_score']
        )
        
        # Calculate ESG Risk Index (inverse relationship: higher score = lower risk)
        df_copy['esg_risk_index'] = 100 - df_copy['composite_esg_score']
        
        print(f"  Weights applied: E={weights['E']}, S={weights['S']}, G={weights['G']}")
        print(f"  Composite ESG Score: {df_copy['composite_esg_score'].mean():.2f} (mean)")
        print(f"  ESG Risk Index: {df_copy['esg_risk_index'].mean():.2f} (mean)")
        print("✓ Weighted ESG scores calculated")
        
        return df_copy
    
    def perform_esg_clustering(self, df, n_clusters=4):
        """
        Cluster companies into ESG performance groups (Leaders, Followers, Laggards)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        n_clusters : int
            Number of clusters
        
        Returns:
        --------
        pd.DataFrame : DataFrame with cluster assignments
        """
        print("\n" + "="*60)
        print("ESG CLUSTERING ANALYSIS")
        print("="*60)
        
        df_copy = df.copy()
        
        # Select ESG-related features for clustering
        cluster_features = ['environment_score', 'social_score', 'governance_score', 
                           'composite_esg_score']
        
        X_cluster = df_copy[cluster_features].dropna()
        
        # Perform K-Means clustering
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.clustering_model.fit_predict(X_cluster)
        
        df_copy.loc[X_cluster.index, 'esg_cluster'] = clusters
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_cluster, clusters)
        
        # Label clusters based on average ESG score
        cluster_stats = df_copy.groupby('esg_cluster')['composite_esg_score'].mean().sort_values(ascending=False)
        cluster_labels = {
            cluster_stats.index[0]: 'ESG Leaders',
            cluster_stats.index[1]: 'ESG Performers',
            cluster_stats.index[2]: 'ESG Followers',
            cluster_stats.index[3]: 'ESG Laggards'
        }
        
        df_copy['esg_cluster_label'] = df_copy['esg_cluster'].map(cluster_labels)
        
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Silhouette score: {silhouette_avg:.3f}")
        print(f"\n  Cluster distribution:")
        for label, count in df_copy['esg_cluster_label'].value_counts().items():
            print(f"    {label}: {count} companies")
        
        print("✓ Clustering complete")
        
        return df_copy
    
    def detect_esg_anomalies(self, df, contamination=0.1):
        """
        Detect ESG anomalies using Isolation Forest
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        contamination : float
            Expected proportion of outliers
        
        Returns:
        --------
        pd.DataFrame : DataFrame with anomaly flags
        """
        print("\n" + "="*60)
        print("ESG ANOMALY DETECTION")
        print("="*60)
        
        df_copy = df.copy()
        
        # Select features for anomaly detection
        anomaly_features = ['environment_score', 'social_score', 'governance_score',
                           'carbon_emissions', 'controversy_score']
        
        X_anomaly = df_copy[[col for col in anomaly_features if col in df_copy.columns]].dropna()
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_predictions = self.anomaly_detector.fit_predict(X_anomaly)
        anomaly_scores = self.anomaly_detector.score_samples(X_anomaly)
        
        df_copy.loc[X_anomaly.index, 'is_anomaly'] = (anomaly_predictions == -1).astype(int)
        df_copy.loc[X_anomaly.index, 'anomaly_score'] = anomaly_scores
        
        n_anomalies = (anomaly_predictions == -1).sum()
        
        print(f"  Contamination rate: {contamination}")
        print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(X_anomaly)*100:.1f}%)")
        print("✓ Anomaly detection complete")
        
        return df_copy
    
    def apply_rule_based_risk_flags(self, df):
        """
        Apply domain-specific rule-based risk flags
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame : DataFrame with risk flags
        """
        print("\n" + "="*60)
        print("APPLYING RULE-BASED RISK FLAGS")
        print("="*60)
        
        df_copy = df.copy()
        
        # Rule 1: High carbon emissions (top 25%)
        if 'carbon_emissions' in df_copy.columns:
            threshold = df_copy['carbon_emissions'].quantile(0.75)
            df_copy['high_carbon_flag'] = (df_copy['carbon_emissions'] > threshold).astype(int)
            n_flagged = df_copy['high_carbon_flag'].sum()
            print(f"  Rule 1 - High Carbon Emissions: {n_flagged} companies flagged")
        
        # Rule 2: High controversy score (≥3)
        if 'controversy_score' in df_copy.columns:
            df_copy['high_controversy_flag'] = (df_copy['controversy_score'] >= 3).astype(int)
            n_flagged = df_copy['high_controversy_flag'].sum()
            print(f"  Rule 2 - High Controversy: {n_flagged} companies flagged")
        
        # Rule 3: Sensitive sectors (Fossil Fuels, Weapons, Tobacco)
        sensitive_sectors = ['Energy', 'Defense', 'Tobacco']
        if 'sector' in df_copy.columns:
            df_copy['sensitive_sector_flag'] = df_copy['sector'].isin(sensitive_sectors).astype(int)
            n_flagged = df_copy['sensitive_sector_flag'].sum()
            print(f"  Rule 3 - Sensitive Sectors: {n_flagged} companies flagged")
        
        # Rule 4: Low governance score (<50)
        if 'governance_score' in df_copy.columns:
            df_copy['low_governance_flag'] = (df_copy['governance_score'] < 50).astype(int)
            n_flagged = df_copy['low_governance_flag'].sum()
            print(f"  Rule 4 - Low Governance: {n_flagged} companies flagged")
        
        # Rule 5: Poor overall ESG (composite score <40)
        if 'composite_esg_score' in df_copy.columns:
            df_copy['poor_esg_flag'] = (df_copy['composite_esg_score'] < 40).astype(int)
            n_flagged = df_copy['poor_esg_flag'].sum()
            print(f"  Rule 5 - Poor Overall ESG: {n_flagged} companies flagged")
        
        # Aggregate risk score (sum of all flags)
        flag_cols = [col for col in df_copy.columns if col.endswith('_flag')]
        df_copy['total_risk_flags'] = df_copy[flag_cols].sum(axis=1)
        
        print(f"\n  Companies with ≥3 risk flags: {(df_copy['total_risk_flags'] >= 3).sum()}")
        print("✓ Risk flags applied")
        
        return df_copy
    
    def train_gradient_boosting_model(self, df, target='composite_esg_score'):
        """
        Train advanced Gradient Boosting model for ESG prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe
        target : str
            Target variable
        
        Returns:
        --------
        dict : Model and performance metrics
        """
        print("\n" + "="*60)
        print("TRAINING AI-ENHANCED ESG PREDICTION MODEL")
        print("="*60)
        
        # Prepare features
        exclude_cols = ['ticker', 'company_name', 'sector', 'industry', 'country',
                       target, 'esg_cluster', 'esg_cluster_label', 'esg_category',
                       'is_anomaly', 'anomaly_score']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number]).dropna()
        y = df.loc[X.index, target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Gradient Boosting model
        print("\n  Training Gradient Boosting Regressor...")
        self.esg_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        
        self.esg_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.esg_model.predict(X_train)
        y_test_pred = self.esg_model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.esg_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Model Performance:")
        print(f"    Training R²: {metrics['train_r2']:.4f}")
        print(f"    Testing R²: {metrics['test_r2']:.4f}")
        print(f"    Testing RMSE: {metrics['test_rmse']:.4f}")
        print(f"    Testing MAE: {metrics['test_mae']:.4f}")
        
        print(f"\n  Top 5 Important Features:")
        for idx, row in self.feature_importance.head().iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        print("\n✓ Model training complete")
        
        return {
            'model': self.esg_model,
            'metrics': metrics,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'actuals': {'train': y_train, 'test': y_test}
        }
    
    def generate_comprehensive_esg_assessment(self, df):
        """
        Complete AI-enhanced ESG assessment pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input processed dataframe
        
        Returns:
        --------
        pd.DataFrame : Fully assessed dataframe with all enhancements
        """
        print("\n" + "="*70)
        print(" AI-ENHANCED ESG RISK ASSESSMENT PIPELINE ".center(70, "="))
        print("="*70)
        
        # Step 1: Calculate weighted ESG scores
        df = self.calculate_weighted_esg_score(df)
        
        # Step 2: Perform clustering
        df = self.perform_esg_clustering(df, n_clusters=4)
        
        # Step 3: Detect anomalies
        df = self.detect_esg_anomalies(df, contamination=0.1)
        
        # Step 4: Apply rule-based risk flags
        df = self.apply_rule_based_risk_flags(df)
        
        # Step 5: Train predictive model
        model_results = self.train_gradient_boosting_model(df, target='composite_esg_score')
        
        print("\n" + "="*70)
        print("✓ AI-ENHANCED ESG ASSESSMENT COMPLETE")
        print("="*70)
        
        return df, model_results
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance from trained model"""
        if self.feature_importance is None:
            print("Please train model first")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top Features for ESG Risk Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/ai_enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Load processed data
    try:
        processed_data = pd.read_csv('data/processed/esg_processed_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run preprocessing first.")
        exit()
    
    # Initialize AI-enhanced model
    ai_model = AIEnhancedESGModel()
    
    # Run comprehensive assessment
    enhanced_data, model_results = ai_model.generate_comprehensive_esg_assessment(processed_data)
    
    # Save enhanced data
    enhanced_data.to_csv('data/processed/esg_ai_enhanced.csv', index=False)
    print("\n✓ Enhanced ESG data saved to 'data/processed/esg_ai_enhanced.csv'")
    
    # Plot feature importance
    ai_model.plot_feature_importance(top_n=15)
    
    # Display sample results
    print("\n" + "="*60)
    print("SAMPLE ENHANCED ESG ASSESSMENTS")
    print("="*60)
    display_cols = ['ticker', 'composite_esg_score', 'esg_risk_index', 
                   'esg_cluster_label', 'total_risk_flags', 'is_anomaly']
    print(enhanced_data[display_cols].head(10).to_string(index=False))
