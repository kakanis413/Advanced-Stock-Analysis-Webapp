#Data Fetcher

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

class DataFetcher:
    """Handles all data fetching operations for stocks and market data"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    
    def get_stock_data(self, symbol, period='1y'):
        """Fetch stock OHLCV data using yfinance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        # Validates required columns: Open, High, Low, Close, Volume
        return data
    
    def get_multiple_stocks_data(self, symbols, period='1y'):
        """Batch fetch data for multiple stocks"""
        # Returns dictionary: {symbol: DataFrame}
        
    def get_stock_info(self, symbol):
        """Get detailed fundamental information"""
        # Returns company info, financials, ratios
        
    def get_sector_data(self, sector_etfs, period='1y'):
        """Fetch sector ETF data for correlation analysis"""
        sector_mapping = {
            'XLK': 'Technology',     'XLF': 'Financial',
            'XLV': 'Healthcare',     'XLE': 'Energy',
            'XLI': 'Industrial',     'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples', 'XLU': 'Utilities',
            'XLB': 'Materials',      'XLRE': 'Real Estate'
        }
        
    def get_market_data(self, period='1y'):
        """Get market indices (SPY, QQQ, DIA, IWM, VTI)"""
        
    def get_financial_data(self, symbol):
        """Get financial statements and metrics"""
        # Returns: financials, balance_sheet, cash_flow, info
        
    def get_earnings_data(self, symbol):
        """Get earnings data and estimates"""
        # Returns: earnings, quarterly_earnings, calendar
        
    def calculate_returns(self, data, period='daily'):
        """Calculate returns for daily/weekly/monthly periods"""
        
    def get_dividend_data(self, symbol):
        """Get dividend history"""