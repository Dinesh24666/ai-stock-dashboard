import os
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Configuration
st.set_page_config(
    page_title="AI Stock Screener", page_icon="📈", layout="wide"
)
st.title("⚡ Indian Market AI Stock Dashboard")

# 2. Setup API Key (Gemini Free Tier)
GEMINI_API_KEY = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get a free key from Google AI Studio",
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())


# 3. Cached Market Data Fetcher
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info
    return hist, info


# 4. Sidebar Controls
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input(
    "Stock Ticker (NSE: append .NS)", value="ICICIBANK.NS"
)

# 5. Main Analysis Pipeline
if ticker_input:
    hist, info = fetch_stock_data(ticker_input)

    if not hist.empty:
        # Technical Calculations
        hist["SMA_50"] = hist["Close"].rolling(50).mean()
        hist["SMA_200"] = hist["Close"].rolling(200).mean()
        curr_price = hist["Close"].iloc[-1]
        sma50 = hist["SMA_50"].iloc[-1]
        sma200 = hist["SMA_200"].iloc[-1]

        # Fundamental Metrics
        roe = (info.get("returnOnEquity") or 0.12) * 100
        pe = info.get("trailingPE") or 20.0
        de = (info.get("debtToEquity") or 50.0) / 100

        # Composite Scoring (0-100)
        fund_score = min(
            100, max(0, (roe / 25.0) * 60 + ((2.0 - min(de, 2.0)) / 2.0) * 40)
        )
        tech_score = (
            (35 if curr_price > sma50 else 0)
            + (35 if curr_price > sma200 else 0)
            + (30 if sma50 > sma200 else 0)
        )
        composite = round(0.6 * fund_score + 0.4 * tech_score, 1)

        # UI Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{curr_price:.2f}")
        col2.metric("Composite Score", f"{composite}/100")
        col3.metric("ROE", f"{roe:.1f}%")
        col4.metric(
            "Trend Alignment",
            "Bullish" if tech_score >= 70 else "Neutral/Bearish",
        )

        # Interactive Candlestick Chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Price",
                ),
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_50"],
                    line=dict(color="orange", width=1.5),
                    name="50 SMA",
                ),
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_200"],
                    line=dict(color="blue", width=1.5),
                    name="200 SMA",
                ),
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 6. Low-Cost AI Summary Trigger (Auto-Detects Valid Gemini Model)
        st.subheader("🤖 AI Investment Thesis")
        if st.button("Generate AI Analysis"):
            if not GEMINI_API_KEY:
                st.warning("Please enter your Gemini API Key in the sidebar.")
            else:
                prompt = f"""
                Analyze the following Indian stock:
                - Company: {info.get('longName', ticker_input)}
                - Current Price: ₹{curr_price:.2f}
                - P/E: {pe:.1f} | ROE: {roe:.1f}% | Debt/Equity: {de:.2f}
                - Technicals: Above 50 SMA: {curr_price > sma50}, Above 200 SMA: {curr_price > sma200}
                - Calculated Score: {composite}/100

                Provide concise bullet points:
                1. Core Catalysts
                2. Key Risks
                3. Final Verdict (Bullish / Cautious / Bearish)
                """

                with st.spinner("Generating AI commentary..."):
                    try:
                        # Automatically select the first supported model on your key
                        available_models = [
                            m.name
                            for m in genai.list_models()
                            if "generateContent"
                            in m.supported_generation_methods
                        ]
                        target_model = next(
                            (
                                m
                                for m in available_models
                                if "flash" in m.lower()
                            ),
                            available_models[0]
                            if available_models
                            else "gemini-1.5-flash",
                        )

                        model = genai.GenerativeModel(target_model)
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as err:
                        st.error(f"Failed to generate analysis: {err}")
    else:
        st.error("No historical data found for this ticker.")
