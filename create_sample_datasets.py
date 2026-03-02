"""
Sample Dataset Generator for ESG Risk Assessment Project
Creates realistic CSV files for testing without API access
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_directories():
    """Create necessary data directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'results',
        'reports'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def generate_yahoo_finance_data(n_companies=10000):
    """Generate Yahoo Finance ESG data"""
    print("\nGenerating Yahoo Finance ESG data...")
    
    sectors = ['Technology', 'Financial Services', 'Healthcare', 'Consumer Goods', 
               'Energy', 'Industrials', 'Utilities', 'Materials', 'Real Estate']
    
    countries = ['United States', 'United Kingdom', 'Germany', 'Japan', 'China', 
                 'France', 'Canada', 'Australia', 'India', 'South Korea']
    
    # Sample company names and tickers
    tech_companies = [
        ('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corporation'), 
        ('GOOGL', 'Alphabet Inc.'), ('AMZN', 'Amazon.com Inc.'),
        ('META', 'Meta Platforms Inc.'), ('NVDA', 'NVIDIA Corporation'),
        ('TSLA', 'Tesla Inc.'), ('INTC', 'Intel Corporation'),
        ('CSCO', 'Cisco Systems'), ('ORCL', 'Oracle Corporation')
    ]
    
    finance_companies = [
        ('JPM', 'JPMorgan Chase'), ('BAC', 'Bank of America'),
        ('WFC', 'Wells Fargo'), ('GS', 'Goldman Sachs'),
        ('MS', 'Morgan Stanley'), ('C', 'Citigroup')
    ]
    
    healthcare_companies = [
        ('JNJ', 'Johnson & Johnson'), ('PFE', 'Pfizer Inc.'),
        ('UNH', 'UnitedHealth Group'), ('ABBV', 'AbbVie Inc.'),
        ('MRK', 'Merck & Co.'), ('TMO', 'Thermo Fisher Scientific')
    ]
    
    consumer_companies = [
        ('PG', 'Procter & Gamble'), ('KO', 'Coca-Cola Company'),
        ('WMT', 'Walmart Inc.'), ('HD', 'Home Depot'),
        ('NKE', 'Nike Inc.'), ('COST', 'Costco Wholesale')
    ]
    
    energy_companies = [
        ('XOM', 'Exxon Mobil'), ('CVX', 'Chevron Corporation'),
        ('COP', 'ConocoPhillips'), ('SLB', 'Schlumberger'),
        ('EOG', 'EOG Resources'), ('OXY', 'Occidental Petroleum')
    ]
    
    industrial_companies = [
        ('BA', 'Boeing Company'), ('CAT', 'Caterpillar Inc.'),
        ('GE', 'General Electric'), ('MMM', '3M Company'),
        ('HON', 'Honeywell International'), ('UPS', 'United Parcel Service')
    ]
    
    utility_companies = [
        ('NEE', 'NextEra Energy'), ('DUK', 'Duke Energy'),
        ('SO', 'Southern Company'), ('D', 'Dominion Energy'),
        ('AEP', 'American Electric Power'), ('EXC', 'Exelon Corporation')
    ]
    
    all_companies = (tech_companies + finance_companies + healthcare_companies + 
                    consumer_companies + energy_companies + industrial_companies + 
                    utility_companies)
    
    while len(all_companies) < n_companies:
        idx = len(all_companies)
        sector = np.random.choice(sectors)
        all_companies.append((f'COMP{idx}', f'{sector} Company {idx}'))
    
    all_companies = all_companies[:n_companies]
    
    data = []
    
    for i, (ticker, company_name) in enumerate(all_companies):
        if ticker in [c[0] for c in tech_companies]:
            sector = 'Technology'
        elif ticker in [c[0] for c in finance_companies]:
            sector = 'Financial Services'
        elif ticker in [c[0] for c in healthcare_companies]:
            sector = 'Healthcare'
        elif ticker in [c[0] for c in consumer_companies]:
            sector = 'Consumer Goods'
        elif ticker in [c[0] for c in energy_companies]:
            sector = 'Energy'
        elif ticker in [c[0] for c in industrial_companies]:
            sector = 'Industrials'
        elif ticker in [c[0] for c in utility_companies]:
            sector = 'Utilities'
        else:
            sector = np.random.choice(sectors)
        
        industry_map = {
            'Technology': ['Software', 'Hardware', 'Semiconductors', 'IT Services'],
            'Financial Services': ['Banking', 'Insurance', 'Asset Management', 'Investment Banking'],
            'Healthcare': ['Pharmaceuticals', 'Biotechnology', 'Medical Devices', 'Healthcare Services'],
            'Consumer Goods': ['Food & Beverage', 'Retail', 'Apparel', 'Consumer Electronics'],
            'Energy': ['Oil & Gas', 'Renewable Energy', 'Energy Equipment', 'Utilities'],
            'Industrials': ['Aerospace', 'Manufacturing', 'Construction', 'Transportation'],
            'Utilities': ['Electric Utilities', 'Gas Utilities', 'Water Utilities', 'Renewable Utilities']
        }
        industry = np.random.choice(industry_map.get(sector, ['Other']))
        
        country = np.random.choice(countries, p=[0.45, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.04])
        market_cap = np.random.lognormal(mean=10, sigma=2) * 1e9
        
        if sector == 'Technology':
            env_base, soc_base, gov_base = 70, 72, 78
        elif sector == 'Energy':
            env_base, soc_base, gov_base = 45, 55, 65
        elif sector == 'Healthcare':
            env_base, soc_base, gov_base = 68, 75, 72
        elif sector == 'Financial Services':
            env_base, soc_base, gov_base = 62, 65, 80
        else:
            env_base, soc_base, gov_base = 60, 63, 70
        
        environment_score = np.clip(env_base + np.random.normal(0, 8), 20, 95)
        social_score = np.clip(soc_base + np.random.normal(0, 8), 20, 95)
        governance_score = np.clip(gov_base + np.random.normal(0, 8), 20, 95)
        total_esg_score = (environment_score + social_score + governance_score) / 3
        
        if sector == 'Energy':
            controversy_level = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        else:
            controversy_level = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.35, 0.2, 0.1, 0.04, 0.01])
        
        data.append({
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'country': country,
            'market_cap': market_cap,
            'total_esg_score': total_esg_score,
            'environment_score': environment_score,
            'social_score': social_score,
            'governance_score': governance_score,
            'controversy_level': controversy_level
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} companies with Yahoo Finance ESG data")
    return df

def generate_kaggle_esg_data(tickers):
    """Generate Kaggle-style detailed ESG metrics"""
    print("\nGenerating Kaggle ESG dataset...")
    
    data = []
    
    for ticker in tickers:
        carbon_emissions = np.random.lognormal(mean=8, sigma=1.5) * 1000
        water_usage = np.random.lognormal(mean=7, sigma=1.2) * 100
        waste_generation = np.random.lognormal(mean=6, sigma=1.3) * 100
        employee_turnover = np.clip(np.random.normal(15, 5), 3, 35)
        diversity_score = np.clip(np.random.normal(60, 15), 20, 95)
        board_independence = np.clip(np.random.normal(70, 12), 30, 95)
        environment_pillar_score = np.clip(np.random.normal(65, 12), 25, 95)
        social_pillar_score = np.clip(np.random.normal(63, 13), 25, 95)
        governance_pillar_score = np.clip(np.random.normal(68, 11), 30, 95)
        controversy_score = np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.30, 0.20, 0.10, 0.05])
        
        data.append({
            'ticker': ticker,
            'environment_pillar_score': environment_pillar_score,
            'social_pillar_score': social_pillar_score,
            'governance_pillar_score': governance_pillar_score,
            'carbon_emissions': carbon_emissions,
            'water_usage': water_usage,
            'waste_generation': waste_generation,
            'employee_turnover': employee_turnover,
            'diversity_score': diversity_score,
            'board_independence': board_independence,
            'controversy_score': controversy_score
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated Kaggle ESG metrics for {len(df)} companies")
    return df

def generate_world_bank_data():
    """Generate World Bank country-level indicators"""
    print("\nGenerating World Bank indicators...")
    
    countries_data = {
        'United States': {
            'co2_emissions_per_capita': 15.2,
            'renewable_energy_pct': 12.6,
            'forest_area_pct': 33.9,
            'corruption_index': 67,
            'human_development_index': 0.926
        },
        'United Kingdom': {
            'co2_emissions_per_capita': 5.4,
            'renewable_energy_pct': 13.8,
            'forest_area_pct': 13.1,
            'corruption_index': 77,
            'human_development_index': 0.932
        },
        'Germany': {
            'co2_emissions_per_capita': 8.5,
            'renewable_energy_pct': 17.4,
            'forest_area_pct': 32.7,
            'corruption_index': 80,
            'human_development_index': 0.947
        },
        'Japan': {
            'co2_emissions_per_capita': 8.7,
            'renewable_energy_pct': 7.6,
            'forest_area_pct': 68.5,
            'corruption_index': 73,
            'human_development_index': 0.925
        },
        'China': {
            'co2_emissions_per_capita': 7.4,
            'renewable_energy_pct': 12.4,
            'forest_area_pct': 23.0,
            'corruption_index': 42,
            'human_development_index': 0.768
        },
        'France': {
            'co2_emissions_per_capita': 4.6,
            'renewable_energy_pct': 11.7,
            'forest_area_pct': 31.5,
            'corruption_index': 69,
            'human_development_index': 0.903
        },
        'Canada': {
            'co2_emissions_per_capita': 15.4,
            'renewable_energy_pct': 18.9,
            'forest_area_pct': 38.7,
            'corruption_index': 77,
            'human_development_index': 0.936
        },
        'Australia': {
            'co2_emissions_per_capita': 16.8,
            'renewable_energy_pct': 7.1,
            'forest_area_pct': 16.2,
            'corruption_index': 75,
            'human_development_index': 0.951
        },
        'India': {
            'co2_emissions_per_capita': 1.9,
            'renewable_energy_pct': 9.8,
            'forest_area_pct': 24.1,
            'corruption_index': 40,
            'human_development_index': 0.645
        },
        'South Korea': {
            'co2_emissions_per_capita': 12.3,
            'renewable_energy_pct': 2.7,
            'forest_area_pct': 63.9,
            'corruption_index': 62,
            'human_development_index': 0.925
        }
    }
    
    data = []
    for country, metrics in countries_data.items():
        data.append({
            'country': country,
            **metrics
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated World Bank data for {len(df)} countries")
    return df

def merge_all_datasets(yahoo_df, kaggle_df, worldbank_df):
    """Merge all datasets"""
    print("\nMerging all datasets...")
    
    merged = pd.merge(yahoo_df, kaggle_df, on='ticker', how='left', suffixes=('_yahoo', '_kaggle'))
    merged = pd.merge(merged, worldbank_df, on='country', how='left')
    
    print(f"✓ Merged dataset created with {len(merged)} records and {len(merged.columns)} features")
    return merged

def add_missing_values(df, missing_rate=0.08):
    """Introduce realistic missing values"""
    print(f"\nIntroducing {missing_rate*100}% missing values...")
    
    df_with_missing = df.copy()
    
    columns_with_missing = [
        'environment_score', 'social_score', 'governance_score',
        'carbon_emissions', 'water_usage', 'waste_generation',
        'employee_turnover', 'diversity_score', 'board_independence'
    ]
    
    for col in columns_with_missing:
        if col in df_with_missing.columns:
            n_missing = int(len(df_with_missing) * missing_rate)
            missing_indices = np.random.choice(df_with_missing.index, n_missing, replace=False)
            df_with_missing.loc[missing_indices, col] = np.nan
    
    total_missing = df_with_missing.isnull().sum().sum()
    total_values = df_with_missing.shape[0] * df_with_missing.shape[1]
    actual_rate = total_missing / total_values
    
    print(f"✓ Introduced {total_missing} missing values ({actual_rate*100:.1f}% of total)")
    return df_with_missing

def main():
    """Generate all sample datasets"""
    print("="*70)
    print(" GENERATING SAMPLE ESG DATASETS ".center(70, "="))
    print("="*70)
    
    create_directories()
    
    yahoo_df = generate_yahoo_finance_data(n_companies=10000)
    kaggle_df = generate_kaggle_esg_data(yahoo_df['ticker'].tolist())
    worldbank_df = generate_world_bank_data()
    
    merged_df = merge_all_datasets(yahoo_df, kaggle_df, worldbank_df)
    raw_df = add_missing_values(merged_df, missing_rate=0.08)
    
    print("\n" + "="*70)
    print("SAVING DATASETS")
    print("="*70)
    
    yahoo_df.to_csv('data/external/yahoo_finance_esg.csv', index=False)
    print(f"✓ Saved: data/external/yahoo_finance_esg.csv ({len(yahoo_df)} rows)")
    
    kaggle_df.to_csv('data/external/kaggle_esg_data.csv', index=False)
    print(f"✓ Saved: data/external/kaggle_esg_data.csv ({len(kaggle_df)} rows)")
    
    worldbank_df.to_csv('data/external/world_bank_indicators.csv', index=False)
    print(f"✓ Saved: data/external/world_bank_indicators.csv ({len(worldbank_df)} rows)")
    
    raw_df.to_csv('data/raw/esg_raw_data.csv', index=False)
    print(f"✓ Saved: data/raw/esg_raw_data.csv ({len(raw_df)} rows, {len(raw_df.columns)} columns)")
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal Companies: {len(raw_df)}")
    print(f"Total Features: {len(raw_df.columns)}")
    print(f"Missing Values: {raw_df.isnull().sum().sum()} ({raw_df.isnull().sum().sum()/(raw_df.shape[0]*raw_df.shape[1])*100:.1f}%)")
    
    print("\nSector Distribution:")
    print(raw_df['sector'].value_counts().to_string())
    
    print("\n" + "="*70)
    print("✓ ALL DATASETS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nYou can now run the pipeline:")
    print("  python src/data_preprocessing.py")
    print("  python src/baseline_models.py")
    print("  python src/ai_enhanced_esg_model.py")
    print("  streamlit run dashboard/streamlit_dashboard.py")

if __name__ == "__main__":
    main()