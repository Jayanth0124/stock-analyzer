# AI Stock Analyzer & Predictor

A comprehensive Streamlit-based web application for analyzing stocks with AI-powered forecasting, technical indicators, news sentiment analysis, and portfolio management.

## Features

- **Dashboard**: Real-time stock price charts with moving averages
- **Technical Indicators**: RSI, Bollinger Bands, MACD
- **AI Forecast**: Hybrid forecasting using Random Forest and Prophet models with sentiment adjustment
- **News & Sentiment**: Latest market news with sentiment analysis using FinBERT or VADER
- **Portfolio Management**: Track your stock positions and performance
- **Export Reports**: Generate PDF reports, CSV, and Excel exports

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/Jayanth0124/stock-analyzer.git>
   cd stock-analyser
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser to `http://localhost:8501`

## Usage

- Select a stock ticker using the search box
- Navigate through different sections using the sidebar
- Customize periods, intervals, and parameters as needed
- Add stocks to your portfolio for tracking

## Live Demo

[View Live Demo](https://nexttrade.streamlit.app/)

## Requirements

- Python 3.8+
- Streamlit
- yfinance
- plotly
- pandas
- numpy
- scikit-learn
- And other dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
