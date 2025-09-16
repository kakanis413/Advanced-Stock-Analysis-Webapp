# Technical Indicators

import pandas as pd
import numpy as np
from scipy import stats

class TechnicalIndicators:
    """Custom technical indicators for advanced stock analysis"""
    
    # Standard Indicators
    def calculate_sma(self, data, window=20): """Simple Moving Average"""
    def calculate_ema(self, data, window=12): """Exponential Moving Average"""
    def calculate_rsi(self, data, window=14): """Relative Strength Index"""
    def calculate_macd(self, data, fast=12, slow=26, signal=9): """MACD Indicator"""
    def calculate_bollinger_bands(self, data, window=20, std_dev=2): """Bollinger Bands"""
    
    # UNIQUE Custom Indicators
    def calculate_custom_momentum(self, data, window=10):
        """Custom Momentum Oscillator - Combines price momentum with volume momentum"""
        price_momentum = data['Close'].pct_change(window)
        volume_momentum = data['Volume'].pct_change(window)
        custom_momentum = (price_momentum * 0.7) + (volume_momentum * 0.3)
        return pd.Series(stats.zscore(custom_momentum.dropna(), nan_policy='omit'))
    
    def calculate_vwsi(self, data, window=14):
        """Volume-Weighted Strength Index - Custom RSI with volume weighting"""
        price_change = data['Close'].diff()
        vw_gains = ((price_change > 0) * price_change * data['Volume']).rolling(window).sum()
        vw_losses = ((price_change < 0) * abs(price_change) * data['Volume']).rolling(window).sum()
        vw_rs = vw_gains / vw_losses.replace(0, np.nan)
        vwsi = 100 - (100 / (1 + vw_rs))
        return vwsi.fillna(50)
    
    def calculate_risk_adjusted_returns(self, data, window=30):
        """Risk-Adjusted Returns (Sharpe-like ratio)"""
        returns = data['Close'].pct_change()
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std().replace(0, np.nan)
        return (rolling_mean / rolling_std).fillna(0)
    
    def calculate_trend_strength(self, data, window=20):
        """Trend Strength using linear regression slope * r-squared"""
        # Calculates trend consistency using statistical regression
        
    def calculate_volatility_breakout(self, data, window=20, multiplier=2):
        """Volatility Breakout signals when volatility spikes significantly"""
        
    def calculate_momentum_divergence(self, data, window=14):
        """Identifies bullish/bearish divergences between price and momentum"""
        
    def calculate_all_indicators(self, data):
        """Calculate ALL indicators and return comprehensive dictionary"""
        # Returns: SMA, EMA, RSI, MACD, Bollinger Bands, Custom Momentum, 
        # VWSI, Risk-Adjusted Returns, Trend Strength, Volatility Breakout, Divergences