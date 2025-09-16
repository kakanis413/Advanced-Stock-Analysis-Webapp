#Risk Analyzer

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """Advanced risk analysis for stocks and portfolios"""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = 0.02  # Assumed risk-free rate (2%)
    
    def calculate_var(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level
        
        Returns:
            float: VaR value
        """
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level
        
        Returns:
            float: Expected shortfall
        """
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """
        Calculate Sharpe ratio
        
        Args:
            returns (pd.Series): Return series
            risk_free_rate (float): Risk-free rate
        
        Returns:
            float: Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate / 252  # Daily risk-free rate
        
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            float: Maximum drawdown
        """
        if len(prices) == 0:
            return 0
        
        # Calculate cumulative maximum
        cumulative_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - cumulative_max) / cumulative_max
        
        return drawdown.min()
    
    def calculate_beta(self, stock_returns, market_returns):
        """
        Calculate stock beta against market
        
        Args:
            stock_returns (pd.Series): Stock return series
            market_returns (pd.Series): Market return series
        
        Returns:
            float: Beta coefficient
        """
        # Align the series by index
        aligned_stock, aligned_market = stock_returns.align(market_returns, join='inner')
        
        if len(aligned_stock) < 10 or len(aligned_market) < 10:
            return 1.0  # Default beta
        
        # Remove any NaN values
        valid_data = pd.DataFrame({'stock': aligned_stock, 'market': aligned_market}).dropna()
        
        if len(valid_data) < 10:
            return 1.0
        
        # Calculate beta using linear regression
        covariance = valid_data['stock'].cov(valid_data['market'])
        market_variance = valid_data['market'].var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_downside_deviation(self, returns, minimum_acceptable_return=0):
        """
        Calculate downside deviation
        
        Args:
            returns (pd.Series): Return series
            minimum_acceptable_return (float): Minimum acceptable return
        
        Returns:
            float: Downside deviation
        """
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < minimum_acceptable_return] - minimum_acceptable_return
        
        if len(downside_returns) == 0:
            return 0
        
        return np.sqrt(np.mean(downside_returns ** 2))
    
    def calculate_sortino_ratio(self, returns, minimum_acceptable_return=0):
        """
        Calculate Sortino ratio
        
        Args:
            returns (pd.Series): Return series
            minimum_acceptable_return (float): Minimum acceptable return
        
        Returns:
            float: Sortino ratio
        """
        if len(returns) == 0:
            return 0
        
        excess_return = returns.mean() - minimum_acceptable_return
        downside_deviation = self.calculate_downside_deviation(returns, minimum_acceptable_return)
        
        if downside_deviation == 0:
            return 0
        
        return excess_return / downside_deviation * np.sqrt(252)  # Annualized
    
    def calculate_individual_metrics(self, stock_data):
        """
        Calculate risk metrics for individual stocks
        
        Args:
            stock_data (dict): Dictionary of stock data
        
        Returns:
            dict: Individual stock metrics
        """
        individual_metrics = {}
        
        for symbol, data in stock_data.items():
            try:
                if data.empty:
                    continue
                
                # Calculate returns
                returns = data['Close'].pct_change().dropna()
                
                if len(returns) == 0:
                    continue
                
                # Calculate various risk metrics
                metrics = {
                    'expected_return': returns.mean() * 252,  # Annualized
                    'volatility': returns.std() * np.sqrt(252),  # Annualized
                    'var_95': self.calculate_var(returns, 0.95),
                    'var_99': self.calculate_var(returns, 0.99),
                    'expected_shortfall_95': self.calculate_expected_shortfall(returns, 0.95),
                    'expected_shortfall_99': self.calculate_expected_shortfall(returns, 0.99),
                    'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                    'sortino_ratio': self.calculate_sortino_ratio(returns),
                    'max_drawdown': self.calculate_max_drawdown(data['Close']),
                    'downside_deviation': self.calculate_downside_deviation(returns),
                    'skewness': stats.skew(returns.dropna()),
                    'kurtosis': stats.kurtosis(returns.dropna()),
                    'calmar_ratio': returns.mean() * 252 / abs(self.calculate_max_drawdown(data['Close'])) if self.calculate_max_drawdown(data['Close']) != 0 else 0
                }
                
                individual_metrics[symbol] = metrics
                
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {str(e)}")
                continue
        
        return individual_metrics
    
    def calculate_portfolio_metrics(self, stock_data, weights=None):
        """
        Calculate portfolio-level risk metrics
        
        Args:
            stock_data (dict): Dictionary of stock data
            weights (dict): Portfolio weights (if None, equal weights assumed)
        
        Returns:
            dict: Portfolio metrics
        """
        if not stock_data:
            return {}
        
        try:
            # Calculate individual metrics first
            individual_metrics = self.calculate_individual_metrics(stock_data)
            
            if not individual_metrics:
                return {'individual_metrics': {}, 'portfolio_metrics': {}}
            
            # Create returns matrix
            returns_data = {}
            for symbol, data in stock_data.items():
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[symbol] = returns
            
            if not returns_data:
                return {'individual_metrics': individual_metrics, 'portfolio_metrics': {}}
            
            # Create DataFrame of returns
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {'individual_metrics': individual_metrics, 'portfolio_metrics': {}}
            
            # Set equal weights if not provided
            if weights is None:
                n_stocks = len(returns_df.columns)
                weights = {col: 1/n_stocks for col in returns_df.columns}
            
            # Ensure weights sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0, index=returns_df.index)
            for symbol in returns_df.columns:
                if symbol in weights:
                    portfolio_returns += returns_df[symbol] * weights[symbol]
            
            # Calculate portfolio metrics
            portfolio_metrics = {
                'expected_return': portfolio_returns.mean() * 252,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'var_95': self.calculate_var(portfolio_returns, 0.95),
                'var_99': self.calculate_var(portfolio_returns, 0.99),
                'expected_shortfall': self.calculate_expected_shortfall(portfolio_returns, 0.95),
                'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns),
                'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns),
                'max_drawdown': self.calculate_max_drawdown(portfolio_returns.cumsum()),
                'skewness': stats.skew(portfolio_returns.dropna()),
                'kurtosis': stats.kurtosis(portfolio_returns.dropna())
            }
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            portfolio_metrics['avg_correlation'] = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            return {
                'individual_metrics': individual_metrics,
                'portfolio_metrics': portfolio_metrics,
                'correlation_matrix': correlation_matrix.to_dict(),
                'weights_used': weights
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {str(e)}")
            individual_metrics = individual_metrics if 'individual_metrics' in locals() else {}
            return {
                'individual_metrics': individual_metrics,
                'portfolio_metrics': {},
                'error': str(e)
            }
    
    def calculate_diversification_score(self, stock_data):
        """
        Calculate diversification score for the portfolio
        
        Args:
            stock_data (dict): Dictionary of stock data
        
        Returns:
            float: Diversification score (0-1, higher is better)
        """
        if len(stock_data) < 2:
            return 0.0
        
        try:
            # Create returns matrix
            returns_data = {}
            for symbol, data in stock_data.items():
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 20:  # Need sufficient data
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return 0.0
            
            # Create DataFrame and calculate correlation matrix
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty or len(returns_df) < 10:
                return 0.0
            
            correlation_matrix = returns_df.corr()
            
            # Calculate average pairwise correlation
            n = len(correlation_matrix)
            correlation_values = []
            
            for i in range(n):
                for j in range(i+1, n):
                    correlation_values.append(abs(correlation_matrix.iloc[i, j]))
            
            if not correlation_values:
                return 0.0
            
            avg_correlation = np.mean(correlation_values)
            
            # Convert correlation to diversification score
            # Lower correlation = higher diversification
            diversification_score = 1 - avg_correlation
            
            # Apply additional factors
            # Number of stocks factor (more stocks generally better diversification)
            n_stocks = len(stock_data)
            stock_factor = min(1.0, n_stocks / 10)  # Optimal around 10 stocks
            
            # Final score combining both factors
            final_score = (diversification_score * 0.8) + (stock_factor * 0.2)
            
            return max(0.0, min(1.0, float(final_score)))
            
        except Exception as e:
            print(f"Error calculating diversification score: {str(e)}")
            return 0.0
    
    def generate_risk_report(self, stock_data):
        """
        Generate comprehensive risk report
        
        Args:
            stock_data (dict): Dictionary of stock data
        
        Returns:
            dict: Comprehensive risk report
        """
        try:
            # Calculate all metrics
            portfolio_analysis = self.calculate_portfolio_metrics(stock_data)
            diversification_score = self.calculate_diversification_score(stock_data)
            
            # Generate risk assessment
            individual_metrics = portfolio_analysis.get('individual_metrics', {})
            portfolio_metrics = portfolio_analysis.get('portfolio_metrics', {})
            
            # Risk level assessment
            risk_level = "Medium"
            risk_factors = []
            
            if portfolio_metrics:
                volatility = portfolio_metrics.get('volatility', 0)
                sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
                max_drawdown = portfolio_metrics.get('max_drawdown', 0)
                
                if volatility > 0.3:
                    risk_factors.append("High volatility")
                    risk_level = "High"
                elif volatility < 0.15:
                    risk_level = "Low"
                
                if sharpe_ratio < 0.5:
                    risk_factors.append("Low risk-adjusted returns")
                
                if max_drawdown < -0.2:
                    risk_factors.append("Significant historical drawdowns")
                    if risk_level != "High":
                        risk_level = "Medium-High"
            
            if diversification_score < 0.5:
                risk_factors.append("Insufficient diversification")
                if risk_level == "Low":
                    risk_level = "Medium"
            
            # Recommendations
            recommendations = []
            
            if diversification_score < 0.6:
                recommendations.append("Consider adding more uncorrelated assets")
            
            if portfolio_metrics.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Review portfolio allocation to improve risk-adjusted returns")
            
            if portfolio_metrics.get('max_drawdown', 0) < -0.15:
                recommendations.append("Implement risk management strategies to limit drawdowns")
            
            return {
                'individual_metrics': individual_metrics,
                'portfolio_metrics': portfolio_metrics,
                'diversification_score': diversification_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'report_date': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating risk report: {str(e)}")
            return {
                'error': str(e),
                'report_date': pd.Timestamp.now().isoformat()
            }
