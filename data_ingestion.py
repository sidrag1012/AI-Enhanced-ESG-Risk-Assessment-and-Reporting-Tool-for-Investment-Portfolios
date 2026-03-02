"""
Data Ingestion Module for ESG Risk Assessment Tool
Fetches data from Yahoo Finance, Kaggle datasets, and World Bank
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


class ESGDataIngestion:
    """
    Handles all data ingestion from multiple sources
    """

    def __init__(self):
        self.yahoo_data = None
        self.kaggle_esg = None
        self.world_bank = None

    # ---------------------------------------------------------
    # YAHOO FINANCE DATA
    # ---------------------------------------------------------
    def fetch_yahoo_finance_esg(self, tickers):
        print("Fetching Yahoo Finance company & market data...")

        yahoo_data = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                data_dict = {
                    "ticker": ticker,
                    "company_name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": info.get("country", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "beta": info.get("beta", None),
                    "trailing_pe": info.get("trailingPE", None),
                    "revenue": info.get("totalRevenue", None),
                }

                yahoo_data.append(data_dict)
                print(f"✓ {ticker}: Company data retrieved")

            except Exception as e:
                print(f"✗ {ticker}: Error - {str(e)}")
                continue  # continue instead of breaking entire loop

        # Convert list to DataFrame AFTER loop
        self.yahoo_data = pd.DataFrame(yahoo_data)

        print(f"\n✓ Yahoo Finance data collected for {len(self.yahoo_data)} companies")
        return self.yahoo_data

    # ---------------------------------------------------------
    # KAGGLE DATA
    # ---------------------------------------------------------
    def load_kaggle_esg_dataset(self, filepath):
        print(f"\nLoading Kaggle ESG dataset from {filepath}...")

        try:
            self.kaggle_esg = pd.read_csv(filepath)
            print(f"✓ Loaded {len(self.kaggle_esg)} records")
            return self.kaggle_esg

        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            print("Creating sample Kaggle ESG dataset...")
            self.kaggle_esg = self._create_sample_kaggle_data()
            return self.kaggle_esg

    # ---------------------------------------------------------
    # WORLD BANK DATA
    # ---------------------------------------------------------
    def load_world_bank_indicators(self, filepath=None):
        print("\nLoading World Bank indicators...")

        if filepath:
            try:
                self.world_bank = pd.read_csv(filepath)
                print(f"✓ Loaded {len(self.world_bank)} records")
                return self.world_bank
            except FileNotFoundError:
                print(f"✗ File not found: {filepath}")

        print("Creating sample World Bank dataset...")
        self.world_bank = self._create_sample_worldbank_data()
        return self.world_bank

    # ---------------------------------------------------------
    # SAMPLE DATA GENERATORS
    # ---------------------------------------------------------
    def _create_sample_kaggle_data(self):
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "XOM", "CVX"]

        data = {
            "ticker": tickers,
            "environment_pillar_score": np.random.uniform(40, 90, len(tickers)),
            "social_pillar_score": np.random.uniform(45, 85, len(tickers)),
            "governance_pillar_score": np.random.uniform(50, 95, len(tickers)),
            "carbon_emissions": np.random.uniform(1000, 50000, len(tickers)),
            "water_usage": np.random.uniform(100, 5000, len(tickers)),
            "waste_generation": np.random.uniform(500, 10000, len(tickers)),
            "employee_turnover": np.random.uniform(5, 25, len(tickers)),
            "diversity_score": np.random.uniform(30, 80, len(tickers)),
            "board_independence": np.random.uniform(40, 90, len(tickers)),
            "controversy_score": np.random.randint(0, 5, len(tickers)),
        }

        return pd.DataFrame(data)

    def _create_sample_worldbank_data(self):
        countries = [
            "United States",
            "United Kingdom",
            "Germany",
            "Japan",
            "China",
        ]

        data = {
            "country": countries,
            "co2_emissions_per_capita": np.random.uniform(3, 16, len(countries)),
            "renewable_energy_pct": np.random.uniform(10, 45, len(countries)),
            "forest_area_pct": np.random.uniform(15, 35, len(countries)),
            "corruption_index": np.random.uniform(30, 85, len(countries)),
            "human_development_index": np.random.uniform(0.75, 0.95, len(countries)),
        }

        return pd.DataFrame(data)

    # ---------------------------------------------------------
    # MERGE ALL SOURCES
    # ---------------------------------------------------------
    def merge_all_sources(self):
        print("\nMerging all data sources...")

        if self.yahoo_data is None or self.kaggle_esg is None:
            print("✗ Please fetch/load Yahoo and Kaggle data first")
            return None

        merged = pd.merge(
            self.yahoo_data,
            self.kaggle_esg,
            on="ticker",
            how="left",
        )

        if self.world_bank is not None:
            merged = pd.merge(
                merged,
                self.world_bank,
                on="country",
                how="left",
            )

        print(
            f"✓ Merged dataset created with {len(merged)} records and {len(merged.columns)} features"
        )

        return merged

    # ---------------------------------------------------------
    # SAVE DATA
    # ---------------------------------------------------------
    def save_raw_data(self, merged_data, filepath="data/raw/esg_raw_data.csv"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        merged_data.to_csv(filepath, index=False)
        print(f"\n✓ Raw data saved to {filepath}")


# =============================================================
# MAIN EXECUTION
# =============================================================
if __name__ == "__main__":

    ingestion = ESGDataIngestion()

    sample_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "JPM",
        "BAC",
        "WMT",
        "JNJ",
        "PG",
        "XOM",
        "CVX",
        "NEE",
        "DUK",
    ]

    yahoo_df = ingestion.fetch_yahoo_finance_esg(sample_tickers)
    kaggle_df = ingestion.load_kaggle_esg_dataset("data/kaggle_esg_data.csv")
    wb_df = ingestion.load_world_bank_indicators("data/world_bank_indicators.csv")

    merged_data = ingestion.merge_all_sources()

    if merged_data is not None:
        print("\n" + "=" * 60)
        print("DATA INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total companies: {len(merged_data)}")
        print(f"Total features: {len(merged_data.columns)}")
        print("\nSample Data:")
        print(merged_data.head())
        print("\nMissing Values:")
        print(merged_data.isnull().sum())