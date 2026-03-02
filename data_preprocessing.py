"""
Data Preprocessing Pipeline for ESG Risk Assessment
Handles missing values, normalization, outliers, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ESGDataPreprocessor:
    """
    Complete preprocessing pipeline for ESG data
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.imputer = KNNImputer(n_neighbors=5)
        self.processed_data = None
        
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values using multiple strategies
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            'auto', 'mean', 'median', 'knn', 'drop'
        
        Returns:
        --------
        pd.DataFrame : DataFrame with imputed values
        """
        print("\n" + "="*60)
        print("MISSING VALUE HANDLING")
        print("="*60)
        
        df_copy = df.copy()
        missing_summary = df_copy.isnull().sum()
        missing_pct = (missing_summary / len(df_copy)) * 100
        
        print("\nMissing values before imputation:")
        print(missing_summary[missing_summary > 0])
        
        # Separate numeric and categorical columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if strategy == 'auto':
            # Use median for ESG scores (less sensitive to outliers)
            score_cols = [col for col in numeric_cols if 'score' in col.lower()]
            for col in score_cols:
                if df_copy[col].isnull().sum() > 0:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                    print(f"  {col}: Filled with median ({df_copy[col].median():.2f})")
            
            # Use mean for other metrics
            other_numeric = [col for col in numeric_cols if col not in score_cols]
            for col in other_numeric:
                if df_copy[col].isnull().sum() > 0:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    print(f"  {col}: Filled with mean ({df_copy[col].mean():.2f})")
                    
        elif strategy == 'knn':
            # KNN imputation for numeric columns
            df_copy[numeric_cols] = self.imputer.fit_transform(df_copy[numeric_cols])
            print("  Applied KNN imputation for all numeric columns")
            
        elif strategy == 'median':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
            print("  Applied median imputation for all numeric columns")
            
        elif strategy == 'mean':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            print("  Applied mean imputation for all numeric columns")
        
        # Handle categorical columns with mode
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                mode_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'Unknown'
                df_copy[col].fillna(mode_value, inplace=True)
                print(f"  {col}: Filled with mode ('{mode_value}')")
        
        print(f"\n✓ Missing values after imputation: {df_copy.isnull().sum().sum()}")
        return df_copy
    
    def detect_and_handle_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect and handle outliers using IQR or Z-score method
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'iqr' or 'zscore'
        threshold : float
            IQR multiplier (1.5) or Z-score threshold (3)
        
        Returns:
        --------
        pd.DataFrame : DataFrame with outliers handled
        """
        print("\n" + "="*60)
        print("OUTLIER DETECTION AND HANDLING")
        print("="*60)
        
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers at bounds instead of removing
                    df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                    df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
                    outlier_summary[col] = outlier_count
                    
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_copy[col].dropna()))
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap at mean ± 3*std
                    mean_val = df_copy[col].mean()
                    std_val = df_copy[col].std()
                    df_copy.loc[df_copy[col] > mean_val + threshold*std_val, col] = mean_val + threshold*std_val
                    df_copy.loc[df_copy[col] < mean_val - threshold*std_val, col] = mean_val - threshold*std_val
                    outlier_summary[col] = outlier_count
        
        print(f"Method: {method.upper()}")
        print(f"Outliers detected and capped:")
        for col, count in outlier_summary.items():
            print(f"  {col}: {count} outliers")
        
        print(f"\n✓ Total outliers handled: {sum(outlier_summary.values())}")
        return df_copy
    
    def normalize_features(self, df, method='minmax'):
        """
        Normalize numeric features to 0-1 scale
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'minmax' or 'standard'
        
        Returns:
        --------
        pd.DataFrame : Normalized dataframe
        """
        print("\n" + "="*60)
        print("FEATURE NORMALIZATION")
        print("="*60)
        
        df_copy = df.copy()
        
        # Identify columns to normalize (exclude identifiers)
        exclude_cols = ['ticker', 'company_name', 'sector', 'industry', 'country']
        numeric_cols = [col for col in df_copy.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if method == 'minmax':
            df_copy[numeric_cols] = self.scaler.fit_transform(df_copy[numeric_cols])
            print(f"  Applied MinMax scaling (0-1 range)")
        elif method == 'standard':
            scaler = StandardScaler()
            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
            print(f"  Applied Standard scaling (mean=0, std=1)")
        
        print(f"  Normalized {len(numeric_cols)} numeric features")
        print(f"\n✓ Normalization complete")
        return df_copy
    
    def encode_categorical_features(self, df):
        """
        Encode categorical variables using Label Encoding
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame : DataFrame with encoded categories
        """
        print("\n" + "="*60)
        print("CATEGORICAL ENCODING")
        print("="*60)
        
        df_copy = df.copy()
        categorical_cols = ['sector', 'industry', 'country']
        
        for col in categorical_cols:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} unique categories encoded")
        
        print(f"\n✓ Encoded {len(categorical_cols)} categorical features")
        return df_copy
    
    def engineer_esg_features(self, df):
        """
        Create derived ESG features for better model performance
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame : DataFrame with engineered features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df_copy = df.copy()
        
        # 1. Weighted ESG composite score (E=40%, S=30%, G=30%)
        if all(col in df_copy.columns for col in ['environment_score', 'social_score', 'governance_score']):
            df_copy['weighted_esg_score'] = (
                0.40 * df_copy['environment_score'] +
                0.30 * df_copy['social_score'] +
                0.30 * df_copy['governance_score']
            )
            print("  Created: weighted_esg_score (40-30-30 weighting)")
        
        # 2. ESG Risk Level (inverse of score)
        if 'weighted_esg_score' in df_copy.columns:
            df_copy['esg_risk_level'] = 100 - df_copy['weighted_esg_score']
            print("  Created: esg_risk_level (100 - weighted_esg_score)")
        
        # 3. Carbon intensity (if market cap available)
        if 'carbon_emissions' in df_copy.columns and 'market_cap' in df_copy.columns:
            df_copy['carbon_intensity'] = df_copy['carbon_emissions'] / (df_copy['market_cap'] / 1e9 + 1)
            print("  Created: carbon_intensity (emissions per $B market cap)")
        
        # 4. Controversy flag (high risk if controversy > 3)
        if 'controversy_score' in df_copy.columns:
            df_copy['high_controversy_flag'] = (df_copy['controversy_score'] >= 3).astype(int)
            print("  Created: high_controversy_flag (1 if score >= 3)")
        
        # 5. ESG performance category
        if 'weighted_esg_score' in df_copy.columns:
            df_copy['esg_category'] = pd.cut(
                df_copy['weighted_esg_score'],
                bins=[0, 40, 60, 80, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            print("  Created: esg_category (Poor/Fair/Good/Excellent)")
        
        print(f"\n✓ Created {5} engineered features")
        return df_copy
    
    def full_preprocessing_pipeline(self, df, save_path='data/processed/esg_processed_data.csv'):
        """
        Execute complete preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw input dataframe
        save_path : str
            Path to save processed data
        
        Returns:
        --------
        pd.DataFrame : Fully preprocessed dataframe
        """
        print("\n" + "="*70)
        print(" ESG DATA PREPROCESSING PIPELINE ".center(70, "="))
        print("="*70)
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df, strategy='auto')
        
        # Step 2: Detect and handle outliers
        df = self.detect_and_handle_outliers(df, method='iqr', threshold=1.5)
        
        # Step 3: Engineer features (before normalization to preserve interpretability)
        df = self.engineer_esg_features(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Normalize numeric features
        df = self.normalize_features(df, method='minmax')
        
        # Save processed data
        df.to_csv(save_path, index=False)
        print(f"\n{'='*70}")
        print(f"✓ PREPROCESSING COMPLETE")
        print(f"✓ Processed data saved to: {save_path}")
        print(f"✓ Final dataset shape: {df.shape}")
        print(f"{'='*70}\n")
        
        self.processed_data = df
        return df


# Example usage
if __name__ == "__main__":
    # Load raw data (from Step 2)
    try:
        raw_data = pd.read_csv('data/raw/esg_raw_data.csv')
    except FileNotFoundError:
        print("Raw data not found. Please run data_ingestion.py first.")
        # Create sample data for demonstration
        raw_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'company_name': ['Apple Inc.', 'Microsoft', 'Alphabet Inc.'],
            'sector': ['Technology', 'Technology', 'Technology'],
            'environment_score': [75, 80, np.nan],
            'social_score': [70, 75, 78],
            'governance_score': [85, 90, 88],
            'carbon_emissions': [25000, 30000, 28000],
            'market_cap': [2.5e12, 2.3e12, 1.8e12],
            'controversy_score': [1, 2, 0]
        })
    
    # Initialize preprocessor
    preprocessor = ESGDataPreprocessor()
    
    # Run full pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(raw_data)
    
    # Display results
    print(processed_data.head())
    print(f"\nProcessed columns: {list(processed_data.columns)}")
