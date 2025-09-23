# Advanced Stock Analysis Webapp

A comprehensive Streamlit-based web application that provides sophisticated stock analysis tools combining technical indicators, fundamental scoring, sentiment analysis, and advanced risk assessment. This dashboard goes beyond traditional stock analysis by offering unique custom metrics and multi-dimensional insights for informed investment decisions.

# Link to Webapp
https://4a61df09-04cb-4342-9560-e74b503fd9be-00-1zfzawl34k2ek.riker.replit.dev

## 🚀 Features

### 📈 Multi-Stock Comparison Dashboard
- **Real-time stock performance comparison** with normalized percentage changes
- **Interactive performance metrics** showing current prices, returns, and volatility
- **Volume analysis charts** for trading activity insights

### 🔗 Advanced Correlation Analysis  
- **Smart correlation heatmaps** between selected stocks
- **Sector correlation analysis** against major ETFs (XLK, XLF, XLV, XLE, XLI)
- **Correlation insights** highlighting strongest and weakest correlations

### 💭 Sentiment Analysis
- **News headline sentiment analysis** using Natural Language Processing
- **Visual sentiment scoring** with color-coded indicators
- **Sentiment summaries** with intuitive emoji representations

### ⚡ Custom Technical Indicators
Beyond standard indicators, includes unique custom algorithms:
- **Custom Momentum Oscillator** - Combines price and volume momentum
- **Volume-Weighted Strength Index (VWSI)** - RSI enhanced with volume weighting
- **Risk-Adjusted Returns** - Sharpe-like ratio calculations
- **Trend Strength Indicator** - Statistical trend consistency measurement
- **Volatility Breakout Signals** - Identifies significant volatility spikes

### ⚖️ Advanced Risk Assessment
- **Value at Risk (VaR)** calculations at 95% and 99% confidence levels
- **Expected Shortfall** and **Sharpe ratio** analysis  
- **Portfolio diversification scoring** with interactive gauge visualization
- **Maximum drawdown analysis** for downside risk evaluation
- **Risk-return scatter plots** for portfolio optimization insights

### 🏆 Multi-Factor Stock Scoring System
Proprietary scoring algorithm with weighted categories:
- **Technical Score** (25%) - RSI, moving averages, volume, volatility
- **Fundamental Score** (30%) - P/E ratio, debt/equity, ROE, margins, growth
- **Momentum Score** (20%) - Multi-timeframe price and volume momentum  
- **Value Score** (15%) - P/B, P/S, EV/EBITDA, dividend yield
- **Quality Score** (10%) - Liquidity ratios, margins, market cap stability

## 🛠️ Technology Stack

- **Frontend**: Streamlit with interactive Plotly visualizations
- **Data Source**: Yahoo Finance API (yfinance) for real-time market data
- **Analytics**: Pandas, NumPy, SciPy for statistical computations
- **NLP**: TextBlob for sentiment analysis
- **Architecture**: Modular object-oriented design with specialized utility classes

## 📁 Project Structure

```
├── app.py                      # Main Streamlit dashboard application
├── utils/
│   ├── data_fetcher.py         # Market data retrieval and processing
│   ├── technical_indicators.py # Custom technical analysis algorithms  
│   ├── sentiment_analyzer.py   # News sentiment analysis with NLP
│   ├── scoring_system.py       # Multi-factor stock scoring system
│   └── risk_analyzer.py        # Advanced risk metrics and portfolio analysis
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── pyproject.toml             # Project dependencies
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/advanced-stock-analysis-dashboard.git
   cd advanced-stock-analysis-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using the project file:
   ```bash
   pip install streamlit pandas numpy plotly scipy yfinance textblob
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   Open your web browser and navigate to `http://localhost:8501`

## 📊 Usage

1. **Select stocks** using the sidebar multiselect (supports up to 10 stocks)
2. **Choose time period** from 1 month to 5 years for analysis
3. **Navigate through tabs** to explore different analysis perspectives:
   - Multi-Stock Dashboard for performance comparison
   - Correlation Analysis for relationship insights  
   - Sentiment Analysis for market sentiment
   - Technical Indicators for custom technical analysis
   - Risk Assessment for portfolio risk evaluation
   - Stock Scoring for comprehensive stock rankings

## 🎯 Unique Value Propositions

- **Custom Technical Indicators**: Proprietary algorithms not available in standard platforms
- **Multi-Dimensional Analysis**: Combines technical, fundamental, sentiment, and risk analysis
- **Interactive Visualizations**: Dynamic charts with drill-down capabilities
- **Real-Time Data**: Live market data integration with error handling
- **Modular Architecture**: Clean, maintainable code structure for extensibility

## 🔧 Configuration

Optional environment variables for enhanced functionality:
- `ALPHA_VANTAGE_API_KEY`: For enhanced news sentiment data
- `NEWS_API_KEY`: For additional news sources

## 📈 Sample Analysis

The dashboard provides insights such as:
- AAPL shows strong technical momentum (Score: 85/100) with bullish sentiment
- Portfolio diversification score of 78% indicates good risk distribution
- TSLA-GOOGL correlation of 0.65 suggests similar market movements
- Current portfolio VaR of -2.3% at 95% confidence level

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## 🙏 Acknowledgments

- Yahoo Finance for providing market data through their API
- Streamlit team for the excellent web app framework  
- Plotly for interactive visualization capabilities
