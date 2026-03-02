"""
Portfolio-Level ESG Risk Analysis
Aggregates company-level ESG scores to portfolio level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PortfolioESGAnalyzer:
    """
    Analyzes ESG risk at portfolio level with sector and country exposures
    """
    
    def __init__(self):
        self.portfolio_data = None
        self.sector_exposure = None
        self.country_exposure = None
        
    def create_portfolio(self, df, portfolio_holdings):
        """
        Create portfolio from holdings
        
        Parameters:
        -----------
        df : pd.DataFrame
            Company ESG data
        portfolio_holdings : dict
            {ticker: weight} mapping
        
        Returns:
        --------
        pd.DataFrame : Portfolio with ESG metrics
        """
        print("\n" + "="*60)
        print("CREATING PORTFOLIO")
        print("="*60)
        
        # Normalize weights to sum to 1
        total_weight = sum(portfolio_holdings.values())
        normalized_holdings = {k: v/total_weight for k, v in portfolio_holdings.items()}
        
        # Filter companies in portfolio
        portfolio_df = df[df['ticker'].isin(portfolio_holdings.keys())].copy()
        portfolio_df['weight'] = portfolio_df['ticker'].map(normalized_holdings)
        
        print(f"  Total companies in portfolio: {len(portfolio_df)}")
        print(f"  Total weight: {portfolio_df['weight'].sum():.2%}")
        
        self.portfolio_data = portfolio_df
        return portfolio_df
    
    def calculate_portfolio_esg_score(self):
        """
        Calculate weighted average ESG score for portfolio
        
        Returns:
        --------
        dict : Portfolio ESG metrics
        """
        print("\n" + "="*60)
        print("CALCULATING PORTFOLIO ESG METRICS")
        print("="*60)
        
        if self.portfolio_data is None:
            print("Please create portfolio first")
            return None
        
        df = self.portfolio_data
        
        # Weighted average scores
        portfolio_metrics = {
            'portfolio_esg_score': (df['composite_esg_score'] * df['weight']).sum(),
            'portfolio_risk_index': (df['esg_risk_index'] * df['weight']).sum(),
            'portfolio_env_score': (df['environment_score'] * df['weight']).sum(),
            'portfolio_social_score': (df['social_score'] * df['weight']).sum(),
            'portfolio_gov_score': (df['governance_score'] * df['weight']).sum(),
            'weighted_carbon_emissions': (df['carbon_emissions'] * df['weight']).sum(),
            'avg_controversy_level': (df['controversy_score'] * df['weight']).sum(),
            'high_risk_companies_pct': (df['total_risk_flags'] >= 3).sum() / len(df) * 100,
            'anomaly_companies_pct': df['is_anomaly'].sum() / len(df) * 100
        }
        
        print(f"\n  Portfolio ESG Score: {portfolio_metrics['portfolio_esg_score']:.2f}")
        print(f"  Portfolio Risk Index: {portfolio_metrics['portfolio_risk_index']:.2f}")
        print(f"  Environment Score: {portfolio_metrics['portfolio_env_score']:.2f}")
        print(f"  Social Score: {portfolio_metrics['portfolio_social_score']:.2f}")
        print(f"  Governance Score: {portfolio_metrics['portfolio_gov_score']:.2f}")
        print(f"  High Risk Companies: {portfolio_metrics['high_risk_companies_pct']:.1f}%")
        print(f"  Anomaly Companies: {portfolio_metrics['anomaly_companies_pct']:.1f}%")
        
        return portfolio_metrics
    
    def analyze_sector_exposure(self):
        """
        Analyze ESG risk by sector exposure
        
        Returns:
        --------
        pd.DataFrame : Sector-wise ESG analysis
        """
        print("\n" + "="*60)
        print("SECTOR-WISE ESG EXPOSURE ANALYSIS")
        print("="*60)
        
        if self.portfolio_data is None:
            print("Please create portfolio first")
            return None
        
        df = self.portfolio_data
        
        sector_analysis = df.groupby('sector').agg({
            'weight': 'sum',
            'composite_esg_score': 'mean',
            'esg_risk_index': 'mean',
            'total_risk_flags': 'mean',
            'ticker': 'count'
        }).round(2)
        
        sector_analysis.columns = ['Weight', 'Avg_ESG_Score', 'Avg_Risk_Index', 
                                   'Avg_Risk_Flags', 'Num_Companies']
        sector_analysis = sector_analysis.sort_values('Weight', ascending=False)
        
        print(f"\n{sector_analysis.to_string()}")
        
        self.sector_exposure = sector_analysis
        return sector_analysis
    
    def analyze_country_exposure(self):
        """
        Analyze ESG risk by country exposure
        
        Returns:
        --------
        pd.DataFrame : Country-wise ESG analysis
        """
        print("\n" + "="*60)
        print("COUNTRY-WISE ESG EXPOSURE ANALYSIS")
        print("="*60)
        
        if self.portfolio_data is None:
            print("Please create portfolio first")
            return None
        
        df = self.portfolio_data
        
        country_analysis = df.groupby('country').agg({
            'weight': 'sum',
            'composite_esg_score': 'mean',
            'esg_risk_index': 'mean',
            'ticker': 'count'
        }).round(2)
        
        country_analysis.columns = ['Weight', 'Avg_ESG_Score', 'Avg_Risk_Index', 'Num_Companies']
        country_analysis = country_analysis.sort_values('Weight', ascending=False)
        
        print(f"\n{country_analysis.to_string()}")
        
        self.country_exposure = country_analysis
        return country_analysis
    
    def identify_top_esg_contributors(self, top_n=10):
        """
        Identify companies contributing most to portfolio ESG score
        
        Parameters:
        -----------
        top_n : int
            Number of top contributors
        
        Returns:
        --------
        pd.DataFrame : Top ESG contributors
        """
        print("\n" + "="*60)
        print(f"TOP {top_n} ESG CONTRIBUTORS")
        print("="*60)
        
        df = self.portfolio_data.copy()
        df['esg_contribution'] = df['composite_esg_score'] * df['weight']
        
        top_contributors = df.nlargest(top_n, 'esg_contribution')[
            ['ticker', 'company_name', 'weight', 'composite_esg_score', 
             'esg_contribution', 'esg_cluster_label']
        ]
        
        print(f"\n{top_contributors.to_string(index=False)}")
        
        return top_contributors
    
    def identify_high_risk_holdings(self):
        """
        Identify high-risk companies in portfolio
        
        Returns:
        --------
        pd.DataFrame : High-risk holdings
        """
        print("\n" + "="*60)
        print("HIGH-RISK HOLDINGS ANALYSIS")
        print("="*60)
        
        df = self.portfolio_data
        
        high_risk = df[df['total_risk_flags'] >= 3].sort_values('weight', ascending=False)
        
        if len(high_risk) > 0:
            print(f"\n  Found {len(high_risk)} high-risk companies:")
            print(f"\n{high_risk[['ticker', 'company_name', 'weight', 'total_risk_flags', 'esg_risk_index']].to_string(index=False)}")
            
            # Specific risk breakdown
            flag_cols = [col for col in df.columns if col.endswith('_flag') and col != 'total_risk_flags']
            risk_breakdown = high_risk[['ticker'] + flag_cols]
            print(f"\n  Risk Flag Breakdown:")
            print(f"{risk_breakdown.to_string(index=False)}")
        else:
            print("\n  No high-risk companies found (risk flags < 3)")
        
        return high_risk
    
    def compare_portfolios(self, portfolio_a_holdings, portfolio_b_holdings, df):
        """
        Compare ESG metrics between two portfolios
        
        Parameters:
        -----------
        portfolio_a_holdings : dict
            Portfolio A holdings {ticker: weight}
        portfolio_b_holdings : dict
            Portfolio B holdings {ticker: weight}
        df : pd.DataFrame
            Company ESG data
        
        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        print("\n" + "="*60)
        print("PORTFOLIO COMPARISON")
        print("="*60)
        
        # Create Portfolio A
        self.create_portfolio(df, portfolio_a_holdings)
        metrics_a = self.calculate_portfolio_esg_score()
        
        # Create Portfolio B
        self.create_portfolio(df, portfolio_b_holdings)
        metrics_b = self.calculate_portfolio_esg_score()
        
        # Comparison table
        comparison = pd.DataFrame({
            'Metric': metrics_a.keys(),
            'Portfolio_A': metrics_a.values(),
            'Portfolio_B': metrics_b.values()
        })
        
        comparison['Difference'] = comparison['Portfolio_B'] - comparison['Portfolio_A']
        comparison['Pct_Change'] = (comparison['Difference'] / comparison['Portfolio_A'] * 100).round(2)
        
        print("\n" + "="*60)
        print("PORTFOLIO COMPARISON RESULTS")
        print("="*60)
        print(f"\n{comparison.to_string(index=False)}")
        
        return comparison
    
    def plot_portfolio_breakdown(self):
        """Create visualization of portfolio ESG breakdown"""
        if self.portfolio_data is None:
            print("Please create portfolio first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sector allocation
        sector_weights = self.portfolio_data.groupby('sector')['weight'].sum().sort_values(ascending=False)
        axes[0, 0].pie(sector_weights, labels=sector_weights.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Sector Allocation')
        
        # 2. ESG Score distribution
        axes[0, 1].hist(self.portfolio_data['composite_esg_score'], bins=15, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.portfolio_data['composite_esg_score'].mean(), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 1].set_xlabel('ESG Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('ESG Score Distribution')
        axes[0, 1].legend()
        
        # 3. Risk flags by company
        top_10 = self.portfolio_data.nlargest(10, 'weight')
        axes[1, 0].barh(range(len(top_10)), top_10['total_risk_flags'])
        axes[1, 0].set_yticks(range(len(top_10)))
        axes[1, 0].set_yticklabels(top_10['ticker'])
        axes[1, 0].set_xlabel('Number of Risk Flags')
        axes[1, 0].set_title('Risk Flags - Top 10 Holdings')
        axes[1, 0].invert_yaxis()
        
        # 4. ESG pillars comparison
        pillars = ['environment_score', 'social_score', 'governance_score']
        avg_scores = [(self.portfolio_data[p] * self.portfolio_data['weight']).sum() for p in pillars]
        axes[1, 1].bar(['Environment', 'Social', 'Governance'], avg_scores, color=['green', 'blue', 'orange'])
        axes[1, 1].set_ylabel('Weighted Average Score')
        axes[1, 1].set_title('ESG Pillar Scores')
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('results/portfolio_esg_breakdown.png', dpi=300, bbox_inches='tight')
        print("\n✓ Portfolio breakdown visualization saved")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Load enhanced ESG data
    try:
        esg_data = pd.read_csv('data/processed/esg_ai_enhanced.csv')
    except FileNotFoundError:
        print("Enhanced ESG data not found. Please run AI-enhanced model first.")
        exit()
    
    # Initialize analyzer
    analyzer = PortfolioESGAnalyzer()
    
    # Define sample portfolios
    portfolio_a = {
        'AAPL': 0.15,
        'MSFT': 0.15,
        'GOOGL': 0.10,
        'AMZN': 0.10,
        'TSLA': 0.10,
        'JPM': 0.10,
        'JNJ': 0.10,
        'WMT': 0.10,
        'PG': 0.10
    }
    
    portfolio_b = {
        'XOM': 0.20,
        'CVX': 0.15,
        'JPM': 0.15,
        'BAC': 0.15,
        'WMT': 0.10,
        'PG': 0.10,
        'JNJ': 0.15
    }
    
    # Analyze Portfolio A
    print("\n" + "="*70)
    print(" PORTFOLIO A ANALYSIS ".center(70, "="))
    print("="*70)
    
    analyzer.create_portfolio(esg_data, portfolio_a)
    metrics_a = analyzer.calculate_portfolio_esg_score()
    sector_exp = analyzer.analyze_sector_exposure()
    country_exp = analyzer.analyze_country_exposure()
    top_contrib = analyzer.identify_top_esg_contributors(top_n=5)
    high_risk = analyzer.identify_high_risk_holdings()
    
    # Plot breakdown
    analyzer.plot_portfolio_breakdown()
    
    # Compare portfolios
    print("\n" + "="*70)
    print(" COMPARING PORTFOLIO A vs PORTFOLIO B ".center(70, "="))
    print("="*70)
    
    comparison = analyzer.compare_portfolios(portfolio_a, portfolio_b, esg_data)
    
    print("\n✓ Portfolio analysis complete")
