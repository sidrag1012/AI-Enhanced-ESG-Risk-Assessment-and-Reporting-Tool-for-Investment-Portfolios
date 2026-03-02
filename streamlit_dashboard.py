"""
Streamlit Dashboard for ESG Risk Assessment Tool
Interactive visualization and analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI-Enhanced ESG Risk Assessment",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load ESG data"""
    try:
        df = pd.read_csv('data/processed/esg_ai_enhanced.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the preprocessing pipeline first.")
        return None

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🌍 AI-Enhanced ESG Risk Assessment & Reporting Tool</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis View",
        ["📈 Dashboard Overview", "🏢 Company Analysis", "💼 Portfolio Analysis", 
         "⚠️ Risk Assessment", "📊 Comparative Analysis"]
    )
    
    # Page routing
    if page == "📈 Dashboard Overview":
        dashboard_overview(df)
    elif page == "🏢 Company Analysis":
        company_analysis(df)
    elif page == "💼 Portfolio Analysis":
        portfolio_analysis(df)
    elif page == "⚠️ Risk Assessment":
        risk_assessment(df)
    elif page == "📊 Comparative Analysis":
        comparative_analysis(df)

def dashboard_overview(df):
    """Main dashboard with key metrics and visualizations"""
    st.header("📈 ESG Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_esg = df['composite_esg_score'].mean()
        st.metric("Average ESG Score", f"{avg_esg:.1f}", 
                 delta=f"{avg_esg - 60:.1f} vs target (60)")
    
    with col2:
        high_risk_pct = (df['total_risk_flags'] >= 3).sum() / len(df) * 100
        st.metric("High Risk Companies", f"{high_risk_pct:.1f}%",
                 delta=f"-{high_risk_pct:.1f}% (lower is better)", delta_color="inverse")
    
    with col3:
        esg_leaders = (df['esg_cluster_label'] == 'ESG Leaders').sum()
        st.metric("ESG Leaders", esg_leaders,
                 delta=f"{esg_leaders/len(df)*100:.1f}% of total")
    
    with col4:
        anomalies = df['is_anomaly'].sum()
        st.metric("Anomalies Detected", anomalies,
                 delta=f"{anomalies/len(df)*100:.1f}% of total", delta_color="inverse")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ESG Score Distribution")
        fig = px.histogram(df, x='composite_esg_score', nbins=30,
                          title="Distribution of Composite ESG Scores",
                          labels={'composite_esg_score': 'ESG Score'})
        fig.add_vline(x=df['composite_esg_score'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text="Mean")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ESG Performance Clusters")
        cluster_counts = df['esg_cluster_label'].value_counts()
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title="Company Distribution by ESG Performance",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sector analysis
    st.subheader("Sector-wise ESG Performance")
    sector_stats = df.groupby('sector').agg({
        'composite_esg_score': 'mean',
        'esg_risk_index': 'mean',
        'total_risk_flags': 'mean'
    }).round(2).reset_index()
    
    fig = px.bar(sector_stats, x='sector', y='composite_esg_score',
                title="Average ESG Score by Sector",
                labels={'composite_esg_score': 'Avg ESG Score', 'sector': 'Sector'},
                color='composite_esg_score',
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # ESG Pillars comparison
    st.subheader("ESG Pillar Scores Comparison")
    pillar_data = pd.DataFrame({
        'Pillar': ['Environment', 'Social', 'Governance'],
        'Average Score': [
            df['environment_score'].mean(),
            df['social_score'].mean(),
            df['governance_score'].mean()
        ],
        'Weight': [40, 30, 30]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pillar_data['Pillar'], y=pillar_data['Average Score'],
                        name='Average Score', marker_color='lightblue'))
    fig.update_layout(title="ESG Pillar Performance (with weights)",
                     yaxis_title="Score", xaxis_title="ESG Pillar")
    st.plotly_chart(fig, use_container_width=True)

def company_analysis(df):
    """Individual company ESG analysis"""
    st.header("🏢 Individual Company Analysis")
    
    # Company selector
    company_list = df['ticker'].unique()
    selected_company = st.selectbox("Select Company", company_list)
    
    company_data = df[df['ticker'] == selected_company].iloc[0]
    
    # Company profile
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{company_data['company_name']} ({selected_company})")
        st.write(f"**Sector:** {company_data['sector']}")
        st.write(f"**Industry:** {company_data['industry']}")
        st.write(f"**Country:** {company_data['country']}")
        st.write(f"**ESG Cluster:** {company_data['esg_cluster_label']}")
    
    with col2:
        # ESG Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=company_data['composite_esg_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ESG Score"},
            delta={'reference': 60},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # ESG Pillars
    st.subheader("ESG Pillar Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌿 Environment", f"{company_data['environment_score']:.1f}",
                 delta=f"{company_data['environment_score'] - df['environment_score'].mean():.1f} vs avg")
    
    with col2:
        st.metric("👥 Social", f"{company_data['social_score']:.1f}",
                 delta=f"{company_data['social_score'] - df['social_score'].mean():.1f} vs avg")
    
    with col3:
        st.metric("⚖️ Governance", f"{company_data['governance_score']:.1f}",
                 delta=f"{company_data['governance_score'] - df['governance_score'].mean():.1f} vs avg")
    
    # Risk flags
    st.subheader("⚠️ Risk Flag Analysis")
    
    flag_cols = [col for col in df.columns if col.endswith('_flag') and col != 'total_risk_flags']
    active_flags = [col.replace('_flag', '').replace('_', ' ').title() 
                   for col in flag_cols if company_data[col] == 1]
    
    if active_flags:
        st.warning(f"**Active Risk Flags ({len(active_flags)}):** {', '.join(active_flags)}")
    else:
        st.success("✅ No active risk flags for this company")
    
    # Anomaly status
    if company_data['is_anomaly'] == 1:
        st.error(f"🚨 This company has been flagged as an ESG anomaly (Anomaly Score: {company_data['anomaly_score']:.3f})")
    else:
        st.success("✅ No anomaly detected")
    
    # Peer comparison
    st.subheader("Peer Comparison")
    peers = df[df['sector'] == company_data['sector']].nsmallest(10, 'esg_risk_index')
    
    fig = px.bar(peers, x='ticker', y='composite_esg_score',
                title=f"Top 10 ESG Performers in {company_data['sector']}",
                labels={'composite_esg_score': 'ESG Score', 'ticker': 'Company'},
                color='composite_esg_score',
                color_continuous_scale='RdYlGn')
    
    # Highlight selected company
    fig.add_hline(y=company_data['composite_esg_score'], 
                 line_dash="dash", line_color="red",
                 annotation_text=f"{selected_company}")
    
    st.plotly_chart(fig, use_container_width=True)

def portfolio_analysis(df):
    """Portfolio-level ESG analysis"""
    st.header("💼 Portfolio ESG Analysis")
    
    st.write("Create a custom portfolio or select a pre-defined template.")
    
    # Portfolio builder
    portfolio_type = st.radio("Portfolio Type", 
                             ["Custom Portfolio", "Sample Tech-Heavy", "Sample Diversified"])
    
    if portfolio_type == "Custom Portfolio":
        st.subheader("Build Your Portfolio")
        
        # Multi-select companies
        selected_companies = st.multiselect("Select Companies", df['ticker'].unique())
        
        if selected_companies:
            # Weight allocation
            st.write("Allocate Weights (must sum to 100%)")
            weights = {}
            cols = st.columns(min(len(selected_companies), 4))
            
            for idx, company in enumerate(selected_companies):
                with cols[idx % 4]:
                    weights[company] = st.number_input(
                        f"{company}", min_value=0.0, max_value=100.0, 
                        value=100.0/len(selected_companies), step=1.0
                    )
            
            total_weight = sum(weights.values())
            
            if abs(total_weight - 100) > 0.1:
                st.warning(f"⚠️ Weights sum to {total_weight:.1f}%. Please adjust to 100%.")
            else:
                # Normalize weights
                weights = {k: v/100 for k, v in weights.items()}
                analyze_portfolio(df, weights)
    
    elif portfolio_type == "Sample Tech-Heavy":
        weights = {
            'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.20, 
            'AMZN': 0.15, 'TSLA': 0.15
        }
        analyze_portfolio(df, weights)
    
    elif portfolio_type == "Sample Diversified":
        weights = {
            'AAPL': 0.12, 'MSFT': 0.12, 'JPM': 0.12, 
            'JNJ': 0.12, 'WMT': 0.12, 'XOM': 0.10,
            'PG': 0.10, 'NEE': 0.10, 'BAC': 0.10
        }
        analyze_portfolio(df, weights)

def analyze_portfolio(df, weights):
    """Analyze portfolio with given weights"""
    # Filter portfolio companies
    portfolio_df = df[df['ticker'].isin(weights.keys())].copy()
    portfolio_df['weight'] = portfolio_df['ticker'].map(weights)
    
    # Calculate portfolio metrics
    portfolio_esg = (portfolio_df['composite_esg_score'] * portfolio_df['weight']).sum()
    portfolio_risk = (portfolio_df['esg_risk_index'] * portfolio_df['weight']).sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio ESG Score", f"{portfolio_esg:.1f}")
    
    with col2:
        st.metric("Portfolio Risk Index", f"{portfolio_risk:.1f}")
    
    with col3:
        high_risk = (portfolio_df['total_risk_flags'] >= 3).sum()
        st.metric("High-Risk Holdings", f"{high_risk}/{len(portfolio_df)}")
    
    # Holdings table
    st.subheader("Portfolio Holdings")
    holdings_display = portfolio_df[['ticker', 'company_name', 'weight', 
                                     'composite_esg_score', 'total_risk_flags']].copy()
    holdings_display['weight'] = (holdings_display['weight'] * 100).round(2)
    holdings_display.columns = ['Ticker', 'Company', 'Weight (%)', 'ESG Score', 'Risk Flags']
    st.dataframe(holdings_display, use_container_width=True)
    
    # Sector allocation
    col1, col2 = st.columns(2)
    
    with col1:
        sector_weights = portfolio_df.groupby('sector')['weight'].sum().reset_index()
        fig = px.pie(sector_weights, values='weight', names='sector',
                    title="Sector Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ESG pillar scores
        pillar_scores = pd.DataFrame({
            'Pillar': ['Environment', 'Social', 'Governance'],
            'Score': [
                (portfolio_df['environment_score'] * portfolio_df['weight']).sum(),
                (portfolio_df['social_score'] * portfolio_df['weight']).sum(),
                (portfolio_df['governance_score'] * portfolio_df['weight']).sum()
            ]
        })
        
        fig = px.bar(pillar_scores, x='Pillar', y='Score',
                    title="Portfolio ESG Pillar Scores",
                    color='Score', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

def risk_assessment(df):
    """Detailed risk assessment view"""
    st.header("⚠️ ESG Risk Assessment")
    
    # Risk overview
    st.subheader("Risk Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk = (df['total_risk_flags'] >= 3).sum()
        st.metric("High Risk", high_risk, 
                 delta=f"{high_risk/len(df)*100:.1f}% of total",
                 delta_color="inverse")
    
    with col2:
        medium_risk = ((df['total_risk_flags'] >= 1) & (df['total_risk_flags'] < 3)).sum()
        st.metric("Medium Risk", medium_risk,
                 delta=f"{medium_risk/len(df)*100:.1f}% of total")
    
    with col3:
        low_risk = (df['total_risk_flags'] == 0).sum()
        st.metric("Low Risk", low_risk,
                 delta=f"{low_risk/len(df)*100:.1f}% of total")
    
    # Risk heatmap
    st.subheader("Risk Heatmap by Sector")
    
    risk_by_sector = df.groupby('sector').agg({
        'high_carbon_flag': 'sum',
        'high_controversy_flag': 'sum',
        'low_governance_flag': 'sum',
        'poor_esg_flag': 'sum'
    })
    
    fig = px.imshow(risk_by_sector.T,
                   labels=dict(x="Sector", y="Risk Type", color="Count"),
                   title="Risk Distribution by Sector",
                   color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk companies
    st.subheader("High-Risk Companies")
    
    high_risk_df = df[df['total_risk_flags'] >= 3].sort_values('total_risk_flags', ascending=False)
    
    if len(high_risk_df) > 0:
        display_df = high_risk_df[['ticker', 'company_name', 'sector', 
                                   'composite_esg_score', 'total_risk_flags']].head(20)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.success("No high-risk companies identified")
    
    # Anomalies
    st.subheader("ESG Anomalies")
    
    anomalies_df = df[df['is_anomaly'] == 1].sort_values('anomaly_score')
    
    if len(anomalies_df) > 0:
        st.write(f"Detected {len(anomalies_df)} anomalous companies:")
        display_df = anomalies_df[['ticker', 'company_name', 'composite_esg_score', 
                                   'anomaly_score']].head(20)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No anomalies detected")

def comparative_analysis(df):
    """Compare companies or portfolios"""
    st.header("📊 Comparative Analysis")
    
    analysis_type = st.radio("Analysis Type", ["Company Comparison", "Sector Comparison"])
    
    if analysis_type == "Company Comparison":
        st.subheader("Compare Companies")
        
        selected_companies = st.multiselect("Select Companies to Compare", 
                                           df['ticker'].unique(), 
                                           default=df['ticker'].unique()[:5])
        
        if len(selected_companies) > 1:
            comparison_df = df[df['ticker'].isin(selected_companies)]
            
            # Radar chart
            categories = ['environment_score', 'social_score', 'governance_score']
            
            fig = go.Figure()
            
            for company in selected_companies:
                company_data = comparison_df[comparison_df['ticker'] == company].iloc[0]
                values = [company_data[cat] for cat in categories] + [company_data[categories[0]]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Environment', 'Social', 'Governance', 'Environment'],
                    name=company,
                    fill='toself'
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="ESG Pillar Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.subheader("Detailed Comparison")
            compare_cols = ['ticker', 'composite_esg_score', 'esg_risk_index',
                          'environment_score', 'social_score', 'governance_score',
                          'total_risk_flags', 'esg_cluster_label']
            st.dataframe(comparison_df[compare_cols], use_container_width=True)
    
    else:  # Sector Comparison
        st.subheader("Sector-wise Comparison")
        
        sector_stats = df.groupby('sector').agg({
            'composite_esg_score': ['mean', 'std'],
            'esg_risk_index': 'mean',
            'total_risk_flags': 'mean',
            'ticker': 'count'
        }).round(2)
        
        sector_stats.columns = ['Avg ESG Score', 'Std Dev', 'Avg Risk Index', 
                               'Avg Risk Flags', 'Num Companies']
        
        st.dataframe(sector_stats, use_container_width=True)
        
        # Sector boxplot
        fig = px.box(df, x='sector', y='composite_esg_score',
                    title="ESG Score Distribution by Sector",
                    labels={'composite_esg_score': 'ESG Score', 'sector': 'Sector'})
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
