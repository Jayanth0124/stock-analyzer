# AI Stock Analyzer & Predictor — Streamlit (Premium, Full App)

import io
import os
import re # Import regex module
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_searchbox import st_searchbox # Import searchbox
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Analyzer",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Optional Libraries ---
def try_import(module_name):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        # st.sidebar.warning(f"Optional library not found: {module_name}", icon="⚠️")
        return None

prophet_mod = try_import("prophet") or try_import("fbprophet")
transformers = try_import("transformers")
torch = try_import("torch")
vader = try_import("vaderSentiment.vaderSentiment")
reportlab_canvas = try_import("reportlab.pdfgen.canvas")
rl_pagesizes = try_import("reportlab.lib.pagesizes")
rl_utils = try_import("reportlab.lib.utils")
rl_units = try_import("reportlab.lib.units")

# --- UI & Styling ---
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
    }
    .stButton>button {
        border-radius: 0.5rem;
        border: 1px solid #38bdf8;
        background-color: transparent;
        color: #38bdf8;
    }
    .stButton>button:hover {
        background-color: rgba(56, 189, 248, 0.2);
        color: #7dd3fc;
        border-color: #7dd3fc;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border-radius: 0.5rem;
        padding: 1.25rem 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-card .sub {
        font-size: 0.875rem;
        color: #94a3b8;
    }
    .metric-card .kpi {
        font-size: 2.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card .good { color: #4ade80; }
    .metric-card .bad { color: #f87171; }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading & Caching ---

@st.cache_data
def get_ticker_data():
    """
    Loads ticker data. For a real app, load a comprehensive CSV from a file.
    To improve, find a CSV file with columns 'Ticker', 'Name' and load it here.
    Example: return pd.read_csv("data/tickers.csv")
    """
    data = {
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'TATAMOTORS.NS', 'RELIANCE.NS', 'INFY.NS', 'BTC-USD', 'ETH-USD'],
        'Name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com, Inc.', 'Tesla, Inc.', 'Tata Motors Limited', 'Reliance Industries Limited', 'Infosys Limited', 'Bitcoin', 'Ethereum']
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not ticker: return pd.DataFrame()
    try:
        return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False).dropna()
    except Exception as e:
        st.error(f"Data fetch failed for {ticker}: {e}", icon="🚨")
        return pd.DataFrame()

# --- Helper Functions (Indicators, AI, etc.) ---

def sma(series, window): return series.rolling(window).mean()
def ema(series, window): return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame): series = series.squeeze()
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower

@st.cache_data(ttl=300)
def fetch_yf_news(symbol: str, limit: int = 15):
    try:
        t = yf.Ticker(symbol)
        news = sorted(t.news or [], key=lambda x: x.get('providerPublishTime', 0), reverse=True)
        return news[:limit]
    except Exception: return []

# (Other helper functions for AI, PDF report etc. remain the same)
# ...
@st.cache_data(ttl=300)
def analyze_sentences(sentences):
    results = []
    if transformers is not None and torch is not None:
        try:
            pipe = transformers.pipeline("sentiment-analysis", model="ProsusAI/finbert")
            for s in sentences:
                out = pipe(s)[0]
                label = out.get("label", "NEUTRAL").upper()
                score = float(out.get("score", 0.0))
                polarity = score if label == "POSITIVE" else -score if label == "NEGATIVE" else 0.0
                results.append({"text": s, "label": label, "score": score, "polarity": polarity})
            return results
        except Exception: pass
    if vader is not None:
        sia = vader.SentimentIntensityAnalyzer()
        for s in sentences:
            vs = sia.polarity_scores(s)
            compound = vs['compound']
            label = "POSITIVE" if compound > 0.05 else "NEGATIVE" if compound < -0.05 else "NEUTRAL"
            results.append({"text": s, "label": label, "score": abs(compound), "polarity": compound})
    return results

def prep_supervised(df: pd.DataFrame, target_col="Close", lags=10):
    X, y = [], []
    series = df[target_col].values
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

@st.cache_data(ttl=600)
def rf_forecast(df: pd.DataFrame, horizon=30, lags=20, target_col="Close"):
    if len(df) < lags + 30: return None, np.nan
    X, y = prep_supervised(df, target_col=target_col, lags=lags)
    n_train = int(len(X) * 0.8)
    X_train, y_train, X_test, y_test = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    model = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test)) if len(X_test) > 0 else np.nan

    last_window = list(df[target_col].values[-lags:])
    preds = []
    for _ in range(horizon):
        x_in = np.array(last_window[-lags:]).reshape(1, -1)
        p = float(model.predict(x_in)[0])
        preds.append(p)
        last_window.append(p)
    idx = pd.date_range(df.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon)
    return pd.DataFrame({"Forecast_RF": preds}, index=idx), mae

@st.cache_data(ttl=600)
def prophet_forecast(df: pd.DataFrame, horizon=30, target_col="Close"):
    if prophet_mod is None: return None
    try:
        Prophet = prophet_mod.Prophet
        m = Prophet(daily_seasonality=True)
        tmp = df.reset_index()
        ds_col = next((c for c in tmp.columns if str(c).lower() in ["date", "datetime", "time", "index"]), tmp.columns[0])
        d = tmp[[ds_col, target_col]].rename(columns={ds_col: "ds", target_col: "y"})
        m.fit(d)
        future = m.make_future_dataframe(periods=horizon, freq="D")
        fc = m.predict(future)
        return fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    except Exception: return None

def blend_predictions(rf_df, prop_df, sentiment_score: float = 0.0):
    if rf_df is None and prop_df is None: return None
    parts = [df for df in [rf_df, prop_df.rename(columns={"yhat": "Forecast_Prophet"}) if prop_df is not None else None] if df is not None]
    merged = pd.concat(parts, axis=1)
    cols = [c for c in ["Forecast_RF", "Forecast_Prophet"] if c in merged.columns]
    if not cols: return None
    merged["Blend"] = merged[cols].mean(axis=1)
    adj = 1.0 + max(-0.015, min(0.015, sentiment_score * 0.02))
    merged["Blend_Adj"] = merged["Blend"] * adj
    return merged

def save_fig_png(fig: go.Figure, path: str):
    try:
        fig.write_image(path, engine="kaleido", scale=2, width=1000, height=500)
        return True
    except Exception: return False

def build_pdf_report(path, ticker, info, last_price, indicators_snapshot, sentiment_summary, figures_paths):
    if reportlab_canvas is None: return False, "ReportLab not installed"
    try:
        cm = rl_units.cm
        c = reportlab_canvas.Canvas(path, pagesize=rl_pagesizes.A4)
        W, H = rl_pagesizes.A4

        def header():
            c.setFillColorRGB(0.1, 0.1, 0.1)
            c.rect(0, H - 2.5 * cm, W, 2.5 * cm, fill=1)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(2 * cm, H - 1.5 * cm, f"AI Stock Report: {ticker}")
            c.setFont("Helvetica", 10)
            c.drawRightString(W - 2 * cm, H - 2 * cm, datetime.now().strftime("%Y-%m-%d %H:%M"))

        def footer():
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawRightString(W - 2 * cm, 1.5 * cm, "Generated by AI Stock Analyzer")

        header()
        y = H - 4 * cm
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 12)
        name, sector, mcap = info.get("shortName", ticker), info.get("sector", "N/A"), info.get("marketCap", 0)
        c.drawString(2 * cm, y, f"Company: {name}"); y -= 0.6 * cm
        c.drawString(2 * cm, y, f"Sector: {sector}"); y -= 0.6 * cm
        c.drawString(2 * cm, y, f"Market Cap: ${mcap:,.0f}"); y -= 0.6 * cm
        c.drawString(2 * cm, y, f"Last Price: ${last_price:,.2f}"); y -= 1.2 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Key Indicators & Sentiment"); y -= 0.6 * cm
        c.setFont("Helvetica", 11)
        for k, v in indicators_snapshot.items():
            c.drawString(2.3 * cm, y, f"{k}: {v}"); y -= 0.5 * cm
        c.drawString(2.3*cm, y, sentiment_summary);
        footer()

        for p in figures_paths:
            try:
                c.showPage()
                header()
                img = rl_utils.ImageReader(p)
                iw, ih = img.getSize()
                aspect = ih / float(iw)
                width = W - 4 * cm
                height = width * aspect
                c.drawImage(img, 2 * cm, H - 3 * cm - height, width=width, height=height)
                footer()
            except Exception: continue

        c.save()
        return True, None
    except Exception as e: return False, str(e)


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

# Load ticker data for searchbox
ticker_df = get_ticker_data()

def search_tickers(search_term: str):
    """Function to search for tickers based on a search term."""
    if not search_term:
        # Show some popular stocks by default
        return ticker_df.head(5).apply(lambda row: f"{row['Name']} ({row['Ticker']})", axis=1).tolist()
    
    # Filter by substring match on both name and ticker, case-insensitive
    mask = ticker_df.apply(
        lambda row: search_term.lower() in row['Name'].lower() or search_term.lower() in row['Ticker'].lower(), 
        axis=1
    )
    # Format for display and return as a list
    return ticker_df[mask].head(10).apply(lambda row: f"{row['Name']} ({row['Ticker']})", axis=1).tolist()

# --- NEW: Searchbox for Ticker Input ---
selected_value = st_searchbox(
    search_tickers,
    key="stock_searchbox",
    placeholder="Search for a stock (e.g., Apple or .NS)",
    default="Apple Inc. (AAPL)",
)

# Parse the selected value to extract the ticker symbol
ticker_match = re.search(r'\((\S+)\)', selected_value or "")
if ticker_match:
    ticker = ticker_match.group(1)
else:
    ticker = "AAPL" # Default fallback

# --- Original Controls ---
c1, c2 = st.sidebar.columns(2)
period = c1.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"], index=6)
interval = c2.selectbox("Interval", ["1d","1wk","1mo"], index=0)
autorefresh_ms = st.sidebar.slider("Auto-refresh (ms)", 0, 120_000, 0, 500, help="Set to 0 to disable.")


# --- Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Technical Indicators", "AI Forecast", "News & Sentiment", "My Portfolio", "Export Report"])

# --- App State & Data Loading ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Quantity", "AvgPrice", "Added"])
if autorefresh_ms > 0:
    st.markdown(f'<meta http-equiv="refresh" content="{autorefresh_ms/1000}">', unsafe_allow_html=True)

data = fetch_history(ticker, period, interval)
if data.empty:
    st.error(f"No data for '{ticker}'. Check the symbol or try a different period/interval.", icon="⚠️")
    st.stop()

close_prices = data['Close'].squeeze()
volume = data['Volume'].squeeze()


# --- Main App ---
st.title(f"📈 AI Stock Analyzer: {ticker}")

# The rest of your app logic (if/elif page == "...") remains the same
# ... (All the page-rendering code from the previous version goes here)
# --- Dashboard Page ---
if page == "Dashboard":
    st.header("Dashboard")
    # KPIs
    last_close = close_prices.iloc[-1]
    prev_close = close_prices.iloc[-2] if len(close_prices) > 1 else last_close
    chg, pct = last_close - prev_close, (last_close - prev_close) / prev_close * 100 if prev_close != 0 else 0.0
    last_vol = float(volume.iloc[-1]) if not volume.empty else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='metric-card'><div class='sub'>Last Price</div><div class='kpi'>${last_close:,.2f}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'><div class='sub'>Change</div><div class='kpi {'good' if chg>=0 else 'bad'}'>{chg:,.2f} ({pct:.2f}%)</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'><div class='sub'>Volume</div><div class='kpi'>{last_vol:,.0f}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'><div class='sub'>Updated</div><div class='kpi'>{datetime.now().strftime('%H:%M:%S')}</div></div>", unsafe_allow_html=True)

    # Price Chart
    st.subheader("Price Chart")
    show_ma = st.checkbox("Show Moving Averages", True)
    if show_ma:
        mc1, mc2 = st.columns(2)
        ma_fast = mc1.slider("Fast MA", 5, 50, 20)
        ma_slow = mc2.slider("Slow MA", 20, 200, 50)
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=close_prices, name="Price")])
    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=sma(close_prices, ma_fast), name=f"SMA {ma_fast}", mode='lines', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=sma(close_prices, ma_slow), name=f"SMA {ma_slow}", mode='lines', line=dict(color='purple')))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- Technical Indicators Page ---
elif page == "Technical Indicators":
    st.header("Technical Indicators")
    c1, c2 = st.columns(2)
    with c1:
        rsi_len = st.slider("RSI Period", 5, 30, 14, key="rsi_slider")
        r = rsi(close_prices, rsi_len)
        fig_rsi = go.Figure(go.Scatter(x=r.index, y=r, name=f"RSI {rsi_len}"))
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.08)", line_width=0)
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rsi, use_container_width=True)
    with c2:
        bb_window = st.slider("Bollinger Bands Window", 10, 50, 20)
        u, m, l = bollinger_bands(close_prices, window=bb_window)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index, y=close_prices, name="Close", line=dict(color='white')))
        fig_bb.add_trace(go.Scatter(x=u.index, y=u, name="Upper Band", line=dict(color='cyan', width=1)))
        fig_bb.add_trace(go.Scatter(x=m.index, y=m, name="Middle Band", line=dict(color='yellow', dash='dot')))
        fig_bb.add_trace(go.Scatter(x=l.index, y=l, name="Lower Band", fill='tonexty', fillcolor='rgba(0,150,255,0.1)', line=dict(color='cyan', width=1)))
        fig_bb.update_layout(title="Bollinger Bands", height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bb, use_container_width=True)
    
    macd_line, sig_line, hist = macd(close_prices)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD", line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color='orange')))
    fig_macd.add_trace(go.Bar(x=hist.index, y=hist, name="Histogram", marker_color=['#4ade80' if v >= 0 else '#f87171' for v in hist]))
    fig_macd.update_layout(title="MACD", height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_macd, use_container_width=True)

# --- AI Forecast Page ---
elif page == "AI Forecast":
    st.header("Hybrid AI Forecast")
    horizon = st.slider("Forecast Horizon (days)", 7, 120, 30)
    daily_close = close_prices.resample('D').agg('last').interpolate()
    with st.spinner("🧠 Running AI models... This may take a moment."):
        rf_df, mae_val = rf_forecast(daily_close.to_frame(name='Close'), horizon=horizon)
        prop_df = prophet_forecast(daily_close.to_frame(name='Close'), horizon=horizon)
    news_items = fetch_yf_news(ticker, limit=12)
    sentiments = analyze_sentences([n.get('title', '') for n in news_items])
    sentiment_score = np.mean([s['polarity'] for s in sentiments]) if sentiments else 0.0
    blended = blend_predictions(rf_df, prop_df, sentiment_score)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=daily_close.index, y=daily_close, name="Historical Close", line=dict(color='white')))
    if blended is not None:
        fig_f.add_trace(go.Scatter(x=blended.index, y=blended['Blend_Adj'], name="Blended (Sentiment Adj.)", line=dict(color='#38bdf8', width=3)))
    if prop_df is not None:
        fig_f.add_trace(go.Scatter(x=prop_df.index, y=prop_df['yhat_lower'], fill=None, mode='lines', line_color='rgba(56,189,248,0.2)'))
        fig_f.add_trace(go.Scatter(x=prop_df.index, y=prop_df['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(56,189,248,0.2)'))
    
    fig_f.update_layout(title=f"{ticker} Forecast ({horizon} days)", height=520, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_f, use_container_width=True)

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("RF MAE", f"{mae_val:.3f}" if not np.isnan(mae_val) else "N/A", help="Mean Absolute Error of the Random Forest model on test data.")
    colm2.metric("Sentiment Score", f"{sentiment_score:+.3f}", help="Aggregate news sentiment. Positive is bullish.")
    active_models = [name for name, model in [("RF", rf_df), ("Prophet", prop_df)] if model is not None]
    colm3.metric("Models Active", " + ".join(active_models) if active_models else "None")

# --- News & Sentiment Page ---
elif page == "News & Sentiment":
    st.header("Market News & Sentiment")
    news_items = fetch_yf_news(ticker, limit=20)
    if not news_items:
        st.info("No news available from yfinance at the moment.")
    else:
        sentiments = analyze_sentences([n.get('title', '') for n in news_items])
        score = np.mean([s['polarity'] for s in sentiments]) if sentiments else 0.0
        sentiment_label = 'Bullish' if score > 0.05 else ('Bearish' if score < -0.05 else 'Neutral')
        st.metric("Aggregate Sentiment", f"{sentiment_label} ({score:+.3f})")
        
        for i, n in enumerate(news_items):
            with st.container():
                st.subheader(f"[{n.get('title', '')}]({n.get('link', '')})")
                sent = sentiments[i] if i < len(sentiments) else {"label": "NEUTRAL", "score": 0.0}
                lab_cls = sent['label'].lower()
                st.caption(f"{n.get('publisher','')} • {datetime.fromtimestamp(n.get('providerPublishTime',0)).strftime('%b %d, %Y')} • Sentiment: {sent['label'].title()} ({sent['score']:.2f})")
                st.divider()

# --- Portfolio Page ---
elif page == "My Portfolio":
    st.header("My Portfolio")
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.form("add_position", clear_on_submit=True):
            st.subheader("Add/Update Position")
            p1, p2, p3 = st.columns(3)
            tkr = p1.text_input("Ticker", value=ticker)
            qty = p2.number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
            avg = p3.number_input("Average Price", min_value=0.0, value=close_prices.iloc[-1], step=0.01)
            if st.form_submit_button("Submit Position"):
                if tkr:
                    df = st.session_state.portfolio
                    new_pos = pd.DataFrame([[tkr.upper(), qty, avg, datetime.now()]], columns=df.columns)
                    df = df[df.Ticker != tkr.upper()]
                    st.session_state.portfolio = pd.concat([df, new_pos], ignore_index=True)
        st.dataframe(st.session_state.portfolio, use_container_width=True, hide_index=True)
    with c2:
        if not st.session_state.portfolio.empty:
            tickers = st.session_state.portfolio['Ticker'].tolist()
            live_prices = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
            
            live_df = st.session_state.portfolio.copy()
            live_df['LivePrice'] = live_df['Ticker'].map(live_prices)
            live_df['Value'] = live_df['Quantity'] * live_df['LivePrice']
            live_df['Cost'] = live_df['Quantity'] * live_df['AvgPrice']
            live_df['PnL'] = live_df['Value'] - live_df['Cost']
            
            total_val, total_cost, total_pnl = live_df['Value'].sum(), live_df['Cost'].sum(), live_df['PnL'].sum()
            pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            st.metric("Total Portfolio Value", f"${total_val:,.2f}")
            st.metric("Total P/L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")

            fig_alloc = px.pie(live_df, names="Ticker", values="Value", title="Portfolio Allocation", hole=0.4)
            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
            fig_alloc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("Add positions to see live performance and allocation.")

# --- Reports Page ---
elif page == "Export Report":
    st.header("Export Report")
    st.subheader("Generate Report (PDF / CSV / Excel)")
    colr1, colr2 = st.columns(2)
    with colr1:
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                info = yf.Ticker(ticker).info
                r_val = rsi(close_prices).iloc[-1]
                macd_val = macd(close_prices)[0].iloc[-1]
                fig_paths = []
                fig_price = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=close_prices, name="Price")])
                fig_price.update_layout(title="Price Chart", template="plotly_dark")
                if save_fig_png(fig_price, f"{ticker}_price.png"): fig_paths.append(f"{ticker}_price.png")
                
                out_path = f"Report_{ticker}_{datetime.now().strftime('%Y%m%d')}.pdf"
                ok, err = build_pdf_report(
                    out_path, ticker, info, close_prices.iloc[-1],
                    {"RSI (last)": f"{r_val:.2f}", "MACD (last)": f"{macd_val:.4f}"},
                    f"Aggregate sentiment score: {np.mean([s['polarity'] for s in analyze_sentences([n.get('title','') for n in fetch_yf_news(ticker)])]):+.3f}",
                    fig_paths
                )
                if ok:
                    with open(out_path, 'rb') as f:
                        st.download_button("✅ Download PDF Report", f, file_name=out_path, mime="application/pdf", use_container_width=True)
                    for p in fig_paths: os.remove(p)
                else: st.error(f"Failed to build PDF: {err}")
    with colr2:
        st.download_button("Download Price Data (CSV)", data.to_csv().encode('utf-8'), file_name=f"{ticker}_prices.csv", mime="text/csv", use_container_width=True)
        try:
            buff = io.BytesIO()
            with pd.ExcelWriter(buff, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='PriceHistory')
            st.download_button("Download Analysis (Excel)", buff.getvalue(), file_name=f"{ticker}_analysis.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
        except Exception as e: st.warning(f"Excel export failed: {e}")