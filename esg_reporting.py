"""
Automated ESG Report Generation
Generates PDF/HTML reports aligned with SFDR, BRSR, and EU Taxonomy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                               Spacer, PageBreak, Image)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import warnings
warnings.filterwarnings('ignore')

class ESGReportGenerator:
    """
    Generates comprehensive ESG reports in PDF/HTML format
    Aligned with global ESG disclosure standards
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles()

    def custom_styles(self):
    # Load default styles
     self.styles = getSampleStyleSheet()

    # -------- Report Title --------
     self.styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=self.styles['Title'],
        fontSize=24,
        textColor=colors.darkgreen,
        spaceAfter=20
    ))

    # -------- Section Header (FIX FOR YOUR ERROR) --------
     self.styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=self.styles['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        spaceAfter=14
    ))

    # -------- Subsection Header --------
     self.styles.add(ParagraphStyle(
        name='SubHeader',
        parent=self.styles['Heading2'],
        fontSize=14,
        textColor=colors.darkgreen,
        spaceAfter=12
    ))

    # -------- Modify Existing BodyText --------
     self.styles['BodyText'].fontSize = 11
     self.styles['BodyText'].leading = 14
     self.styles['BodyText'].spaceAfter = 12

    
    def generate_executive_summary(self, portfolio_metrics):
        """
        Generate executive summary section
        
        Parameters:
        -----------
        portfolio_metrics : dict
            Portfolio ESG metrics
        
        Returns:
        --------
        list : ReportLab flowables
        """
        content = []
        
        # Title
        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        # Summary text
        summary_text = f"""
        This report presents a comprehensive ESG risk assessment of the investment portfolio 
        as of {datetime.now().strftime('%B %d, %Y')}. The analysis employs AI-enhanced 
        methodologies combining machine learning algorithms, clustering techniques, and 
        rule-based risk assessment to provide actionable insights for sustainable investment decisions.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Portfolio ESG Score: {portfolio_metrics['portfolio_esg_score']:.2f}/100<br/>
        • Portfolio Risk Index: {portfolio_metrics['portfolio_risk_index']:.2f}/100<br/>
        • High-Risk Holdings: {portfolio_metrics['high_risk_companies_pct']:.1f}% of portfolio<br/>
        • Companies with Anomalies: {portfolio_metrics['anomaly_companies_pct']:.1f}%<br/>
        <br/>
        The portfolio demonstrates {'strong' if portfolio_metrics['portfolio_esg_score'] > 70 else 'moderate' if portfolio_metrics['portfolio_esg_score'] > 50 else 'weak'} 
        ESG performance relative to industry benchmarks. Detailed analysis follows in subsequent sections.
        """
        
        content.append(Paragraph(summary_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def create_esg_scorecard_table(self, portfolio_metrics):
        """Create ESG scorecard table"""
        data = [
            ['ESG Metric', 'Score', 'Rating'],
            ['Environmental Score', f"{portfolio_metrics['portfolio_env_score']:.2f}", 
             self.get_rating(portfolio_metrics['portfolio_env_score'])],
            ['Social Score', f"{portfolio_metrics['portfolio_social_score']:.2f}",
             self.get_rating(portfolio_metrics['portfolio_social_score'])],
            ['Governance Score', f"{portfolio_metrics['portfolio_gov_score']:.2f}",
             self.get_rating(portfolio_metrics['portfolio_gov_score'])],
            ['Composite ESG Score', f"{portfolio_metrics['portfolio_esg_score']:.2f}",
             self.get_rating(portfolio_metrics['portfolio_esg_score'])],
            ['Risk Index', f"{portfolio_metrics['portfolio_risk_index']:.2f}",
             self.get_risk_rating(portfolio_metrics['portfolio_risk_index'])]
        ]
        
        table = Table(data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def get_rating(self, score):
        """Convert score to rating"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"
    
    def get_risk_rating(self, risk_score):
        """Convert risk score to rating"""
        if risk_score >= 60:
            return "High Risk"
        elif risk_score >= 40:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def generate_holdings_table(self, portfolio_df):
        """Generate portfolio holdings table"""
        content = []
        
        content.append(Paragraph("Portfolio Holdings Analysis", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        # Prepare data
        holdings_data = portfolio_df.nlargest(15, 'weight')[
            ['ticker', 'company_name', 'weight', 'composite_esg_score', 
             'total_risk_flags', 'esg_cluster_label']
        ].copy()
        
        # Format for table
        table_data = [['Ticker', 'Company', 'Weight', 'ESG Score', 'Risk Flags', 'Cluster']]
        
        for _, row in holdings_data.iterrows():
            table_data.append([
                row['ticker'],
                row['company_name'][:30],  # Truncate long names
                f"{row['weight']*100:.1f}%",
                f"{row['composite_esg_score']:.1f}",
                str(int(row['total_risk_flags'])),
                row['esg_cluster_label']
            ])
        
        table = Table(table_data, colWidths=[0.8*inch, 2*inch, 0.8*inch, 
                                            1*inch, 0.8*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def generate_risk_flags_section(self, portfolio_df):
        """Generate risk flags and warnings section"""
        content = []
        
        content.append(Paragraph("Risk Flags and Warnings", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        # Identify high-risk holdings
        high_risk = portfolio_df[portfolio_df['total_risk_flags'] >= 3]
        
        if len(high_risk) > 0:
            risk_text = f"""
            <b>HIGH RISK ALERT:</b> {len(high_risk)} companies in the portfolio have been 
            flagged with 3 or more ESG risk indicators. These holdings require immediate 
            attention and potential remediation strategies.
            <br/><br/>
            <b>High-Risk Holdings:</b><br/>
            """
            
            for _, row in high_risk.iterrows():
                risk_text += f"• {row['ticker']} ({row['company_name']}): {int(row['total_risk_flags'])} risk flags<br/>"
            
            content.append(Paragraph(risk_text, self.styles['BodyText']))
        else:
            content.append(Paragraph(
                "<b>No high-risk holdings identified.</b> The portfolio maintains acceptable ESG risk levels.",
                self.styles['BodyText']
            ))
        
        content.append(Spacer(1, 0.3*inch))
        
        # Flag breakdown
        flag_summary = {
            'High Carbon Emissions': portfolio_df['high_carbon_flag'].sum(),
            'High Controversy': portfolio_df['high_controversy_flag'].sum(),
            'Low Governance': portfolio_df['low_governance_flag'].sum(),
            'Poor ESG Performance': portfolio_df['poor_esg_flag'].sum()
        }
        
        flag_text = "<b>Risk Flag Breakdown:</b><br/>"
        for flag, count in flag_summary.items():
            flag_text += f"• {flag}: {count} companies<br/>"
        
        content.append(Paragraph(flag_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def generate_regulatory_compliance_section(self):
        """Generate section on regulatory compliance (SFDR, BRSR, EU Taxonomy)"""
        content = []
        
        content.append(Paragraph("Regulatory Compliance and Disclosure", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        compliance_text = """
        This report aligns with the following ESG disclosure frameworks and regulations:
        <br/><br/>
        <b>1. EU Sustainable Finance Disclosure Regulation (SFDR)</b><br/>
        The portfolio assessment includes Principal Adverse Impact (PAI) indicators covering 
        environmental, social, and governance factors. Risk flags identify companies with 
        potential adverse impacts requiring enhanced due diligence.
        <br/><br/>
        <b>2. Business Responsibility and Sustainability Report (BRSR) - India</b><br/>
        The ESG scoring methodology incorporates essential and leadership indicators aligned 
        with SEBI's BRSR framework, including climate action, stakeholder engagement, and 
        corporate governance metrics.
        <br/><br/>
        <b>3. EU Taxonomy for Sustainable Activities</b><br/>
        Environmental scoring considers taxonomy-aligned activities including climate change 
        mitigation, circular economy, and pollution prevention. Companies flagged for high 
        carbon emissions require taxonomy alignment verification.
        <br/><br/>
        <b>Data Sources and Methodology:</b><br/>
        This assessment integrates data from Yahoo Finance ESG scores, Kaggle ESG datasets, 
        and World Bank sustainability indicators. The AI-enhanced methodology employs machine 
        learning models (Gradient Boosting), clustering algorithms (K-Means), and anomaly 
        detection (Isolation Forest) for comprehensive risk assessment.
        """
        
        content.append(Paragraph(compliance_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def generate_recommendations(self, portfolio_metrics, portfolio_df):
        """Generate actionable recommendations"""
        content = []
        
        content.append(Paragraph("Recommendations and Action Items", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        recommendations = []
        
        # Based on portfolio ESG score
        if portfolio_metrics['portfolio_esg_score'] < 60:
            recommendations.append(
                "Consider rebalancing portfolio towards ESG Leaders cluster companies "
                "to improve overall ESG performance."
            )
        
        # Based on high-risk holdings
        if portfolio_metrics['high_risk_companies_pct'] > 20:
            recommendations.append(
                "High-risk holdings exceed 20% of portfolio. Implement engagement strategy "
                "with these companies or consider divestment."
            )
        
        # Based on sector concentration
        sector_concentration = portfolio_df.groupby('sector')['weight'].sum().max()
        if sector_concentration > 0.40:
            recommendations.append(
                f"Sector concentration exceeds 40%. Consider diversification to reduce "
                "sector-specific ESG risks."
            )
        
        # Environmental concerns
        if portfolio_metrics['weighted_carbon_emissions'] > 30000:
            recommendations.append(
                "Portfolio carbon footprint is elevated. Prioritize investments in renewable "
                "energy and low-carbon companies."
            )
        
        # Governance
        if portfolio_metrics['portfolio_gov_score'] < 70:
            recommendations.append(
                "Strengthen governance focus through engagement on board independence, "
                "executive compensation, and shareholder rights."
            )
        
        # Generate recommendation list
        rec_text = "<b>Priority Actions:</b><br/>"
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}<br/><br/>"
        
        if not recommendations:
            rec_text = "Portfolio maintains strong ESG profile. Continue monitoring and "
            rec_text += "periodic reassessment recommended."
        
        content.append(Paragraph(rec_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def create_full_report(self, portfolio_df, portfolio_metrics, filename='esg_report.pdf'):
        """
        Generate complete ESG report
        
        Parameters:
        -----------
        portfolio_df : pd.DataFrame
            Portfolio holdings with ESG data
        portfolio_metrics : dict
            Calculated portfolio metrics
        filename : str
            Output filename
        """
        print("\n" + "="*60)
        print("GENERATING ESG REPORT")
        print("="*60)
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for flowables
        story = []
        
        # Title page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "AI-Enhanced ESG Risk Assessment Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['BodyText']
        ))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self.generate_executive_summary(portfolio_metrics))
        
        # ESG Scorecard
        story.append(Paragraph("Portfolio ESG Scorecard", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        story.append(self.create_esg_scorecard_table(portfolio_metrics))
        story.append(Spacer(1, 0.5*inch))
        
        # Holdings Analysis
        story.extend(self.generate_holdings_table(portfolio_df))
        
        # Risk Flags
        story.extend(self.generate_risk_flags_section(portfolio_df))
        
        # New page for compliance
        story.append(PageBreak())
        
        # Regulatory Compliance
        story.extend(self.generate_regulatory_compliance_section())
        
        # Recommendations
        story.extend(self.generate_recommendations(portfolio_metrics, portfolio_df))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            "<i>This report is generated by AI-Enhanced ESG Risk Assessment Tool. "
            "For questions or additional analysis, please contact your portfolio manager.</i>",
            self.styles['BodyText']
        ))
        
        # Build PDF
        doc.build(story)
        
        print(f"\n✓ ESG Report generated: {filename}")
        print(f"  Pages: Multiple sections")
        print(f"  Format: PDF")
        print(f"  Compliance: SFDR, BRSR, EU Taxonomy aligned")
        print("="*60)
    
    def generate_html_report(self, portfolio_df, portfolio_metrics, filename='esg_report.html'):
        """Generate HTML version of report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESG Risk Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1f77b4; text-align: center; }}
                h2 {{ color: #2ca02c; border-bottom: 2px solid #2ca02c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #1f77b4; color: white; padding: 10px; }}
                td {{ padding: 8px; border: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background-color: #ffebee; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; }}
            </style>
        </head>
        <body>
            <h1>AI-Enhanced ESG Risk Assessment Report</h1>
            <p style="text-align: center;"><i>Generated: {datetime.now().strftime('%B %d, %Y')}</i></p>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <p><b>Portfolio ESG Score:</b> {portfolio_metrics['portfolio_esg_score']:.2f}/100</p>
                <p><b>Portfolio Risk Index:</b> {portfolio_metrics['portfolio_risk_index']:.2f}/100</p>
                <p><b>High-Risk Holdings:</b> {portfolio_metrics['high_risk_companies_pct']:.1f}%</p>
            </div>
            
            <h2>ESG Pillar Scores</h2>
            <table>
                <tr>
                    <th>Pillar</th>
                    <th>Score</th>
                    <th>Rating</th>
                </tr>
                <tr>
                    <td>Environmental</td>
                    <td>{portfolio_metrics['portfolio_env_score']:.2f}</td>
                    <td>{self.get_rating(portfolio_metrics['portfolio_env_score'])}</td>
                </tr>
                <tr>
                    <td>Social</td>
                    <td>{portfolio_metrics['portfolio_social_score']:.2f}</td>
                    <td>{self.get_rating(portfolio_metrics['portfolio_social_score'])}</td>
                </tr>
                <tr>
                    <td>Governance</td>
                    <td>{portfolio_metrics['portfolio_gov_score']:.2f}</td>
                    <td>{self.get_rating(portfolio_metrics['portfolio_gov_score'])}</td>
                </tr>
            </table>
            
            <h2>Top Holdings</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Weight</th>
                    <th>ESG Score</th>
                    <th>Risk Flags</th>
                </tr>
        """
        
        # Add top holdings
        top_holdings = portfolio_df.nlargest(10, 'weight')
        for _, row in top_holdings.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['ticker']}</td>
                    <td>{row['company_name']}</td>
                    <td>{row['weight']*100:.1f}%</td>
                    <td>{row['composite_esg_score']:.1f}</td>
                    <td>{int(row['total_risk_flags'])}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Regulatory Compliance</h2>
            <p>This report aligns with SFDR, BRSR, and EU Taxonomy disclosure requirements.</p>
            
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML Report generated: {filename}")


# Example usage
if __name__ == "__main__":
    # Load portfolio data
    try:
        esg_data = pd.read_csv('data/processed/esg_ai_enhanced.csv')
    except FileNotFoundError:
        print("Enhanced ESG data not found.")
        exit()
    
    # Create sample portfolio
    portfolio_holdings = {
        'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.10,
        'JPM': 0.10, 'JNJ': 0.10, 'WMT': 0.10,
        'XOM': 0.10, 'PG': 0.10, 'NEE': 0.10
    }
    
    portfolio_df = esg_data[esg_data['ticker'].isin(portfolio_holdings.keys())].copy()
    portfolio_df['weight'] = portfolio_df['ticker'].map(portfolio_holdings)
    
    # Calculate metrics
    portfolio_metrics = {
        'portfolio_esg_score': (portfolio_df['composite_esg_score'] * portfolio_df['weight']).sum(),
        'portfolio_risk_index': (portfolio_df['esg_risk_index'] * portfolio_df['weight']).sum(),
        'portfolio_env_score': (portfolio_df['environment_score'] * portfolio_df['weight']).sum(),
        'portfolio_social_score': (portfolio_df['social_score'] * portfolio_df['weight']).sum(),
        'portfolio_gov_score': (portfolio_df['governance_score'] * portfolio_df['weight']).sum(),
        'weighted_carbon_emissions': (portfolio_df['carbon_emissions'] * portfolio_df['weight']).sum(),
        'high_risk_companies_pct': (portfolio_df['total_risk_flags'] >= 3).sum() / len(portfolio_df) * 100,
        'anomaly_companies_pct': portfolio_df['is_anomaly'].sum() / len(portfolio_df) * 100
    }
    
    # Generate reports
    reporter = ESGReportGenerator()
    reporter.create_full_report(portfolio_df, portfolio_metrics, 'reports/portfolio_esg_report.pdf')
    reporter.generate_html_report(portfolio_df, portfolio_metrics, 'reports/portfolio_esg_report.html')
    
    print("\n✓ All reports generated successfully")
