#App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.scoring_system import StockScorer
from utils.risk_analyzer import RiskAnalyzer

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = ['AAPL', 'GOOGL', 'MSFT']
if 'time_period' not in st.session_state:
    st.session_state.time_period = '1y'

def main():
    st.title("🚀 Advanced Stock Analysis Dashboard")
    st.markdown("*Unique stock analysis with custom metrics and advanced visualizations*")
    
    # Initialize utilities
    data_fetcher = DataFetcher()
    tech_indicators = TechnicalIndicators()
    sentiment_analyzer = SentimentAnalyzer()
    stock_scorer = StockScorer()
    risk_analyzer = RiskAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("📊 Dashboard Configuration")
    
    # Stock selection
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Analysis",
        options=default_stocks + ['SPY', 'QQQ', 'VTI', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG'],
        default=st.session_state.selected_stocks,
        max_selections=10
    )
    
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        st.rerun()
    
    # Time period selection
    time_periods = {
        '1mo': '1 Month',
        '3mo': '3 Months', 
        '6mo': '6 Months',
        '1y': '1 Year',
        '2y': '2 Years',
        '5y': '5 Years'
    }
    
    selected_period = st.sidebar.selectbox(
        "Analysis Time Period",
        options=list(time_periods.keys()),
        format_func=lambda x: time_periods[x],
        index=list(time_periods.keys()).index(st.session_state.time_period)
    )
    
    if selected_period != st.session_state.time_period:
        st.session_state.time_period = selected_period
        st.rerun()
    
    # Fetch data for selected stocks
    if not selected_stocks:
        st.warning("⚠️ Please select at least one stock to analyze.")
        return
    
    with st.spinner("🔄 Fetching stock data and performing analysis..."):
        # Get stock data
        stock_data = {}
        for symbol in selected_stocks:
            try:
                data = data_fetcher.get_stock_data(symbol, selected_period)
                if not data.empty:
                    stock_data[symbol] = data
            except Exception as e:
                st.error(f"❌ Failed to fetch data for {symbol}: {str(e)}")
        
        if not stock_data:
            st.error("❌ No valid stock data available. Please check your selections.")
            return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Multi-Stock Dashboard", 
        "🔗 Correlation Analysis", 
        "💭 Sentiment Analysis",
        "⚡ Technical Indicators",
        "⚖️ Risk Assessment",
        "🏆 Stock Scoring"
    ])
    
    with tab1:
        st.header("📈 Multi-Stock Comparison Dashboard")
        
        # Performance overview
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Price comparison chart
            fig = go.Figure()
            
            for symbol, data in stock_data.items():
                # Normalize prices to show percentage change
                normalized = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="📊 Stock Performance Comparison (% Change)",
                xaxis_title="Date",
                yaxis_title="Percentage Change (%)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance metrics table
            st.subheader("📋 Performance Metrics")
            
            metrics_data = []
            for symbol, data in stock_data.items():
                current_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                total_return = ((current_price - start_price) / start_price) * 100
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                metrics_data.append({
                    'Symbol': symbol,
                    'Current Price': f"${current_price:.2f}",
                    'Total Return': f"{total_return:.2f}%",
                    'Volatility': f"{volatility:.2f}%"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True)
        
        # Volume analysis
        st.subheader("📊 Volume Analysis")
        
        fig_volume = go.Figure()
        
        for symbol, data in stock_data.items():
            fig_volume.add_trace(go.Scatter(
                x=data.index,
                y=data['Volume'],
                mode='lines',
                name=f"{symbol} Volume",
                line=dict(width=1, dash='dot')
            ))
        
        fig_volume.update_layout(
            title="Trading Volume Comparison",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        st.header("🔗 Correlation Analysis")
        
        if len(stock_data) < 2:
            st.warning("⚠️ Need at least 2 stocks for correlation analysis.")
        else:
            # Calculate correlation matrix
            returns_data = {}
            for symbol, data in stock_data.items():
                returns_data[symbol] = data['Close'].pct_change().dropna()
            
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Correlation heatmap
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="📊 Stock Returns Correlation Matrix",
                    color_continuous_scale='RdBu_r'
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Correlation Insights")
                
                # Find highest and lowest correlations
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        stock1 = correlation_matrix.columns[i]
                        stock2 = correlation_matrix.columns[j]
                        corr_value = correlation_matrix.iloc[i, j]
                        corr_pairs.append((stock1, stock2, corr_value))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                st.write("**Highest Correlations:**")
                for i, (s1, s2, corr) in enumerate(corr_pairs[:3]):
                    st.write(f"{i+1}. {s1} - {s2}: {corr:.3f}")
                
                st.write("**Lowest Correlations:**")
                for i, (s1, s2, corr) in enumerate(corr_pairs[-3:]):
                    st.write(f"{i+1}. {s1} - {s2}: {corr:.3f}")
        
        # Sector correlation (if applicable)
        st.subheader("🏢 Sector Analysis")
        
        sector_data = data_fetcher.get_sector_data(['XLK', 'XLF', 'XLV', 'XLE', 'XLI'], selected_period)
        
        if sector_data:
            sector_returns = {}
            for sector, data in sector_data.items():
                sector_returns[sector] = data['Close'].pct_change().dropna()
            
            stock_returns = {}
            for symbol, data in stock_data.items():
                stock_returns[symbol] = data['Close'].pct_change().dropna()
            
            # Calculate correlation between stocks and sectors
            sector_corr_data = []
            for stock_symbol, stock_ret in stock_returns.items():
                for sector_symbol, sector_ret in sector_returns.items():
                    # Align dates
                    common_dates = stock_ret.index.intersection(sector_ret.index)
                    if len(common_dates) > 30:  # Need sufficient data
                        corr = stock_ret.loc[common_dates].corr(sector_ret.loc[common_dates])
                        sector_corr_data.append({
                            'Stock': stock_symbol,
                            'Sector': sector_symbol,
                            'Correlation': corr
                        })
            
            if sector_corr_data:
                sector_corr_df = pd.DataFrame(sector_corr_data)
                pivot_df = sector_corr_df.pivot(index='Stock', columns='Sector', values='Correlation')
                
                fig_sector_corr = px.imshow(
                    pivot_df,
                    text_auto=True,
                    aspect="auto",
                    title="📊 Stock-Sector Correlation Matrix",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_sector_corr, use_container_width=True)
    
    with tab3:
        st.header("💭 Sentiment Analysis")
        
        # Get sentiment data for selected stocks
        sentiment_data = {}
        
        for symbol in selected_stocks:
            try:
                sentiment = sentiment_analyzer.get_stock_sentiment(symbol)
                sentiment_data[symbol] = sentiment
            except Exception as e:
                st.warning(f"⚠️ Could not fetch sentiment for {symbol}: {str(e)}")
        
        if sentiment_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sentiment scores chart
                sentiment_df = pd.DataFrame.from_dict(sentiment_data, orient='index')
                
                fig_sentiment = go.Figure()
                
                fig_sentiment.add_trace(go.Bar(
                    x=sentiment_df.index,
                    y=sentiment_df['compound_score'],
                    name='Sentiment Score',
                    marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' 
                                for x in sentiment_df['compound_score']]
                ))
                
                fig_sentiment.update_layout(
                    title="📊 Stock Sentiment Analysis",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Sentiment Score (-1 to 1)",
                    yaxis=dict(range=[-1, 1]),
                    height=400
                )
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                st.subheader("📝 Sentiment Summary")
                
                for symbol, sentiment in sentiment_data.items():
                    score = sentiment['compound_score']
                    if score > 0.1:
                        emoji = "😊"
                        label = "Positive"
                    elif score < -0.1:
                        emoji = "😟"
                        label = "Negative"
                    else:
                        emoji = "😐"
                        label = "Neutral"
                    
                    st.write(f"**{symbol}** {emoji} {label} ({score:.3f})")
                    
                    if 'news_count' in sentiment:
                        st.write(f"   📰 News articles: {sentiment['news_count']}")
        else:
            st.info("ℹ️ Sentiment analysis data not available for selected stocks.")
    
    with tab4:
        st.header("⚡ Custom Technical Indicators")
        
        selected_stock_tech = st.selectbox(
            "Select Stock for Technical Analysis",
            options=list(stock_data.keys()),
            key="tech_stock_select"
        )
        
        if selected_stock_tech:
            data = stock_data[selected_stock_tech]
            
            # Calculate custom technical indicators
            indicators = tech_indicators.calculate_all_indicators(data)
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f"{selected_stock_tech} Price & Moving Averages",
                    "Custom Momentum Oscillator",
                    "Volume-Weighted Strength Index",
                    "Risk-Adjusted Returns"
                ),
                row_width=[0.3, 0.2, 0.2, 0.3]
            )
            
            # Price and moving averages
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'],
                mode='lines', name='Close Price',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['SMA_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['EMA_12'],
                mode='lines', name='EMA 12',
                line=dict(color='red', width=1)
            ), row=1, col=1)
            
            # Custom momentum oscillator
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['Custom_Momentum'],
                mode='lines', name='Custom Momentum',
                line=dict(color='purple', width=2)
            ), row=2, col=1)
            
            # Volume-weighted strength index
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['VWSI'],
                mode='lines', name='VWSI',
                line=dict(color='green', width=2)
            ), row=3, col=1)
            
            # Risk-adjusted returns
            fig.add_trace(go.Scatter(
                x=data.index, y=indicators['Risk_Adjusted_Returns'],
                mode='lines', name='Risk-Adjusted Returns',
                line=dict(color='brown', width=2)
            ), row=4, col=1)
            
            fig.update_layout(height=800, title_text=f"📊 Technical Analysis Dashboard - {selected_stock_tech}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical summary
            st.subheader("📋 Technical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_momentum = indicators['Custom_Momentum'].iloc[-1]
                st.metric("Custom Momentum", f"{current_momentum:.4f}",
                         delta=f"{current_momentum - indicators['Custom_Momentum'].iloc[-2]:.4f}")
            
            with col2:
                current_vwsi = indicators['VWSI'].iloc[-1]
                st.metric("VWSI", f"{current_vwsi:.2f}",
                         delta=f"{current_vwsi - indicators['VWSI'].iloc[-2]:.2f}")
            
            with col3:
                current_rar = indicators['Risk_Adjusted_Returns'].iloc[-1]
                st.metric("Risk-Adj Returns", f"{current_rar:.4f}",
                         delta=f"{current_rar - indicators['Risk_Adjusted_Returns'].iloc[-2]:.4f}")
            
            with col4:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Annualized Volatility", f"{volatility:.2f}%")
    
    with tab5:
        st.header("⚖️ Risk Assessment & Portfolio Analysis")
        
        if len(stock_data) < 2:
            st.warning("⚠️ Need at least 2 stocks for portfolio risk analysis.")
        else:
            # Calculate portfolio metrics
            risk_metrics = risk_analyzer.calculate_portfolio_metrics(stock_data)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Risk Metrics")
                
                # Risk metrics table
                risk_df = pd.DataFrame.from_dict(risk_metrics['individual_metrics'], orient='index')
                st.dataframe(risk_df.round(4))
                
                # Portfolio-level metrics
                st.subheader("📋 Portfolio Summary")
                portfolio_metrics = risk_metrics['portfolio_metrics']
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Portfolio VaR (95%)", f"{portfolio_metrics['var_95']:.4f}")
                    st.metric("Expected Shortfall", f"{portfolio_metrics['expected_shortfall']:.4f}")
                
                with metric_col2:
                    st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.4f}")
                    st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.4f}")
            
            with col2:
                # Risk-return scatter plot
                fig_risk_return = go.Figure()
                
                for symbol in risk_metrics['individual_metrics'].keys():
                    metrics = risk_metrics['individual_metrics'][symbol]
                    fig_risk_return.add_trace(go.Scatter(
                        x=[metrics['volatility']],
                        y=[metrics['expected_return']],
                        mode='markers+text',
                        text=[symbol],
                        textposition='top center',
                        marker=dict(size=12),
                        name=symbol
                    ))
                
                fig_risk_return.update_layout(
                    title="📊 Risk-Return Profile",
                    xaxis_title="Risk (Volatility)",
                    yaxis_title="Expected Return",
                    height=400
                )
                
                st.plotly_chart(fig_risk_return, use_container_width=True)
                
                # Diversification analysis
                st.subheader("🎯 Diversification Score")
                
                diversification_score = risk_analyzer.calculate_diversification_score(stock_data)
                
                # Create a gauge chart for diversification score
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = diversification_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diversification Score (%)"},
                    delta = {'reference': 80},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab6:
        st.header("🏆 Custom Stock Scoring System")
        
        # Calculate scores for all stocks
        stock_scores = {}
        
        for symbol, data in stock_data.items():
            try:
                # Get additional data needed for scoring
                info = yf.Ticker(symbol).info
                score = stock_scorer.calculate_comprehensive_score(data, info)
                stock_scores[symbol] = score
            except Exception as e:
                st.warning(f"⚠️ Could not calculate score for {symbol}: {str(e)}")
        
        if stock_scores:
            # Create scoring dashboard
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Scores breakdown chart
                score_categories = ['technical_score', 'fundamental_score', 'momentum_score', 
                                  'value_score', 'quality_score']
                
                fig_scores = go.Figure()
                
                for category in score_categories:
                    scores = [stock_scores[symbol].get(category, 0) for symbol in stock_scores.keys()]
                    fig_scores.add_trace(go.Bar(
                        name=category.replace('_', ' ').title(),
                        x=list(stock_scores.keys()),
                        y=scores
                    ))
                
                fig_scores.update_layout(
                    title="📊 Stock Scoring Breakdown",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                st.subheader("🏅 Stock Rankings")
                
                # Calculate overall scores and rank
                overall_scores = []
                for symbol, scores in stock_scores.items():
                    overall_score = scores.get('overall_score', 0)
                    overall_scores.append({
                        'Symbol': symbol,
                        'Overall Score': overall_score,
                        'Rank': 0  # Will be filled after sorting
                    })
                
                # Sort by overall score
                overall_scores.sort(key=lambda x: x['Overall Score'], reverse=True)
                
                # Add ranks
                for i, score_data in enumerate(overall_scores):
                    score_data['Rank'] = i + 1
                
                # Display rankings
                rankings_df = pd.DataFrame(overall_scores)
                rankings_df['Overall Score'] = rankings_df['Overall Score'].round(2)
                
                st.dataframe(rankings_df, hide_index=True)
            
            # Detailed scoring explanation
            st.subheader("📝 Scoring Methodology")
            
            st.write("""
            **Our Custom Scoring System evaluates stocks across 5 key dimensions:**
            
            1. **Technical Score**: Based on momentum, trend strength, and technical indicators
            2. **Fundamental Score**: P/E ratio, debt-to-equity, ROE, and financial health metrics
            3. **Momentum Score**: Recent price performance and volume trends
            4. **Value Score**: Price-to-book, price-to-sales, and valuation metrics
            5. **Quality Score**: Profit margins, revenue growth, and business quality indicators
            
            Each dimension is scored from 0-100, and the overall score is a weighted average.
            """)
            
            # Show detailed scores for selected stock
            selected_stock_detail = st.selectbox(
                "Select Stock for Detailed Scoring",
                options=list(stock_scores.keys()),
                key="detail_stock_select"
            )
            
            if selected_stock_detail:
                detailed_score = stock_scores[selected_stock_detail]
                
                st.subheader(f"📊 Detailed Analysis: {selected_stock_detail}")
                
                score_col1, score_col2, score_col3 = st.columns(3)
                
                with score_col1:
                    st.metric("Technical Score", f"{detailed_score.get('technical_score', 0):.1f}/100")
                    st.metric("Momentum Score", f"{detailed_score.get('momentum_score', 0):.1f}/100")
                
                with score_col2:
                    st.metric("Fundamental Score", f"{detailed_score.get('fundamental_score', 0):.1f}/100")
                    st.metric("Value Score", f"{detailed_score.get('value_score', 0):.1f}/100")
                
                with score_col3:
                    st.metric("Quality Score", f"{detailed_score.get('quality_score', 0):.1f}/100")
                    st.metric("**Overall Score**", f"**{detailed_score.get('overall_score', 0):.1f}/100**")
        
        else:
            st.info("ℹ️ Stock scoring data not available. Please check your stock selections.")

if __name__ == "__main__":
    main()
