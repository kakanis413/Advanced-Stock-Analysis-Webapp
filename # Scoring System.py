# Scoring System

mport pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class StockScorer:
    """Custom stock scoring system based on multiple factors"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.25,      # 25% weight
            'fundamental': 0.30,    # 30% weight  
            'momentum': 0.20,       # 20% weight
            'value': 0.15,          # 15% weight
            'quality': 0.10         # 10% weight
        }
    
    def calculate_technical_score(self, data):
        """Calculate technical analysis score (RSI, Moving Averages, Volume, Volatility)"""
        # RSI Score - optimal range 30-70
        # Moving Average Score - price above MAs, uptrend signals
        # Volume Score - higher recent volume preferred
        # Volatility Score - moderate volatility (15-35%) preferred
        
    def calculate_fundamental_score(self, info):
        """Calculate fundamental analysis score (P/E, Debt/Equity, ROE, Margins, Growth)"""
        # P/E Ratio Score - optimal 10-25 range
        # Debt to Equity Score - lower is better
        # ROE Score - >15% preferred
        # Profit Margin Score - >10% good, >20% excellent
        # Revenue Growth Score - >5% good, >15% excellent
        
    def calculate_momentum_score(self, data):
        """Calculate momentum score (1mo, 3mo returns, acceleration, volume momentum)"""
        # Price momentum across multiple timeframes
        # Recent acceleration comparing 10-day periods
        # Volume momentum trends
        
    def calculate_value_score(self, info):
        """Calculate value score (P/B, P/S, EV/EBITDA, Dividend Yield)"""
        # Price to Book - lower preferred for value
        # Price to Sales - <2.0 excellent
        # EV/EBITDA - <10 excellent
        # Dividend Yield - moderate 2-6% preferred
        
    def calculate_quality_score(self, info):
        """Calculate quality score (Liquidity, Margins, Market Cap stability)"""
        # Current Ratio - 1.5-3.0 ideal for liquidity
        # Gross Margin - higher is better
        # Operating Margin - >20% excellent
        # Market Cap - larger companies get stability bonus
        
    def calculate_comprehensive_score(self, data, info):
        """Calculate weighted overall score combining all factors"""
        # Returns dict with all individual scores plus weighted overall score