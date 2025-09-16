#Sentiment Analyzer

import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import os
from datetime import datetime, timedelta
import json

class SentimentAnalyzer:
    """Handles sentiment analysis for stocks using news and social media data"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY', 'demo')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    
    def get_news_headlines(self, symbol, days_back=7):
        """
        Fetch recent news headlines for a stock
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
        
        Returns:
            list: List of news headlines
        """
        headlines = []
        
        try:
            # Using Alpha Vantage News API if available
            if self.alpha_vantage_key != 'demo':
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.alpha_vantage_key,
                    'limit': 50
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'feed' in data:
                        for article in data['feed']:
                            headlines.append(article.get('title', ''))
            
            # Fallback: Generate sample headlines for demonstration
            if not headlines:
                headlines = self._generate_sample_headlines(symbol)
                
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            headlines = self._generate_sample_headlines(symbol)
        
        return headlines[:20]  # Limit to 20 headlines
    
    def _generate_sample_headlines(self, symbol):
        """
        Generate sample headlines for demonstration purposes
        Note: In production, this would be replaced with real API calls
        """
        sample_headlines = [
            f"{symbol} reports strong quarterly earnings beat",
            f"Analysts upgrade {symbol} stock rating",
            f"{symbol} announces new product launch",
            f"Market volatility affects {symbol} stock price",
            f"{symbol} CEO discusses growth strategy",
            f"Institutional investors increase {symbol} holdings",
            f"{symbol} faces regulatory challenges",
            f"Technology sector outlook affects {symbol}",
            f"{symbol} stock shows resilient performance",
            f"Market analysts remain bullish on {symbol}"
        ]
        
        return sample_headlines
    
    def analyze_sentiment(self, text_list):
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text_list (list): List of text strings
        
        Returns:
            dict: Sentiment analysis results
        """
        if not text_list:
            return {
                'compound_score': 0.0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total_count': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text in text_list:
            if text and isinstance(text, str):
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                sentiments.append(polarity)
                
                if polarity > 0.1:
                    positive_count += 1
                elif polarity < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        if sentiments:
            compound_score = np.mean(sentiments)
        else:
            compound_score = 0.0
        
        return {
            'compound_score': compound_score,
            'positive': positive_count,
            'neutral': neutral_count,
            'negative': negative_count,
            'total_count': len(text_list)
        }
    
    def get_stock_sentiment(self, symbol):
        """
        Get overall sentiment for a stock
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            dict: Comprehensive sentiment analysis
        """
        try:
            # Get news headlines
            headlines = self.get_news_headlines(symbol)
            
            # Analyze sentiment
            sentiment_results = self.analyze_sentiment(headlines)
            
            # Add additional metadata
            sentiment_results['symbol'] = symbol
            sentiment_results['news_count'] = len(headlines)
            sentiment_results['timestamp'] = datetime.now().isoformat()
            
            # Calculate sentiment strength
            sentiment_results['sentiment_strength'] = abs(sentiment_results['compound_score'])
            
            # Classify sentiment
            compound = sentiment_results['compound_score']
            if compound > 0.1:
                sentiment_results['sentiment_label'] = 'Positive'
            elif compound < -0.1:
                sentiment_results['sentiment_label'] = 'Negative'
            else:
                sentiment_results['sentiment_label'] = 'Neutral'
            
            return sentiment_results
            
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {
                'compound_score': 0.0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total_count': 0,
                'symbol': symbol,
                'news_count': 0,
                'sentiment_strength': 0.0,
                'sentiment_label': 'Neutral',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_market_sentiment(self, symbols_list):
        """
        Get market-wide sentiment analysis
        
        Args:
            symbols_list (list): List of stock symbols
        
        Returns:
            dict: Market sentiment summary
        """
        market_sentiments = {}
        
        for symbol in symbols_list:
            sentiment = self.get_stock_sentiment(symbol)
            market_sentiments[symbol] = sentiment
        
        # Calculate overall market sentiment
        compound_scores = [data['compound_score'] for data in market_sentiments.values()]
        
        if compound_scores:
            market_compound = np.mean(compound_scores)
            market_std = np.std(compound_scores)
        else:
            market_compound = 0.0
            market_std = 0.0
        
        market_summary = {
            'overall_sentiment': market_compound,
            'sentiment_volatility': market_std,
            'positive_stocks': len([s for s in compound_scores if s > 0.1]),
            'negative_stocks': len([s for s in compound_scores if s < -0.1]),
            'neutral_stocks': len([s for s in compound_scores if -0.1 <= s <= 0.1]),
            'total_stocks': len(compound_scores)
        }
        
        return {
            'individual_sentiments': market_sentiments,
            'market_summary': market_summary
        }
    
    def get_sentiment_trend(self, symbol, days=30):
        """
        Get sentiment trend over time (simulated for demonstration)
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to analyze
        
        Returns:
            pd.DataFrame: Sentiment trend data
        """
        # In production, this would fetch historical sentiment data
        # For demonstration, we'll simulate sentiment trend
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate sentiment scores with some randomness and trend
        np.random.seed(hash(symbol) % 1000)  # Consistent randomness for each symbol
        base_sentiment = np.random.uniform(-0.2, 0.2)  # Base sentiment for the stock
        
        sentiment_scores = []
        for i, date in enumerate(dates):
            # Add trend and noise
            trend = (i / days) * 0.1 * np.random.choice([-1, 1])  # Slight trend
            noise = np.random.normal(0, 0.1)  # Random noise
            
            score = base_sentiment + trend + noise
            score = np.clip(score, -1, 1)  # Keep within valid range
            sentiment_scores.append(score)
        
        sentiment_trend_df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'symbol': symbol
        })
        
        # Add rolling averages
        sentiment_trend_df['sentiment_ma_3'] = sentiment_trend_df['sentiment_score'].rolling(3).mean()
        sentiment_trend_df['sentiment_ma_7'] = sentiment_trend_df['sentiment_score'].rolling(7).mean()
        
        return sentiment_trend_df
    
    def calculate_sentiment_momentum(self, symbol, short_window=3, long_window=10):
        """
        Calculate sentiment momentum
        
        Args:
            symbol (str): Stock symbol
            short_window (int): Short-term window
            long_window (int): Long-term window
        
        Returns:
            dict: Sentiment momentum metrics
        """
        trend_data = self.get_sentiment_trend(symbol, days=long_window + 5)
        
        if len(trend_data) >= long_window:
            short_avg = trend_data['sentiment_score'].tail(short_window).mean()
            long_avg = trend_data['sentiment_score'].tail(long_window).mean()
            
            momentum = short_avg - long_avg
            
            return {
                'sentiment_momentum': momentum,
                'short_term_avg': short_avg,
                'long_term_avg': long_avg,
                'momentum_strength': abs(momentum)
            }
        else:
            return {
                'sentiment_momentum': 0.0,
                'short_term_avg': 0.0,
                'long_term_avg': 0.0,
                'momentum_strength': 0.0
            }
