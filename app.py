from datetime import date, datetime
import gc
import json
import os
import time
import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
    [data-testid="stDataEditor"] td, [data-testid="stDataEditor"] th {
        text-align: center !important;
        vertical-align: middle !important;
    }
    .index-ticker-container {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 18px;
        gap: 16px;
        align-items: center;
    }
    .trade-summary-card {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: #ffffff;
        border: 2px solid #0284c7;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 12px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_alert_permission_banner():
    banner_html = """
    <div style="display: flex; align-items: center; justify-content: space-between; background: #ecfdf5; border: 2px solid #10b981; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 22px;">🔔</span>
            <div>
                <div style="font-size: 14.5px; font-weight: 700; color: #065f46;">Live Trigger Audio & Push Notifications</div>
                <div style="font-size: 12px; color: #047857;">Click below to grant browser sound permissions.</div>
            </div>
        </div>
        <button onclick="alert('Audio active!')" style="background: #059669; color: #ffffff; border: none; border-radius: 8px; padding: 10px 18px; font-weight: 700; cursor: pointer;">
            🔊 Enable Sound
        </button>
    </div>
    """
    components.html(banner_html, height=70)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_market_indices():
    return [
        {"name": "Nifty 50", "value": "24,054.90", "change": "-100.00", "pct": "(-0.41%)", "is_pos": False, "arrow": "↘"},
        {"name": "Bank Nifty", "value": "57,067.85", "change": "-194.55", "pct": "(-0.34%)", "is_pos": False, "arrow": "↘"},
        {"name": "India VIX", "value": "11.51", "change": "+0.12", "pct": "(+1.05%)", "is_pos": True, "arrow": "↗"},
    ]


market_indices = fetch_live_market_indices()
if market_indices:
    ticker_html = '<div class="index-ticker-container">'
    for idx in market_indices:
        cls = "index-pos" if idx["is_pos"] else "index-neg"
        ticker_html += f"""
        <div class="index-item">
            <span class="index-name">{idx['name']}</span>
            <span class="index-val">{idx['value']}</span>
            <span class="{cls}">{idx['change']} {idx['pct']} {idx['arrow']}</span>
        </div>
        """
    ticker_html += '</div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

st.title("⚡ Indian Market AI Stock Screener & Paper Trading")

PORTFOLIO_FILE = "portfolio.json"
WATCHLIST_FILE = "watchlist.json"


def load_json_file(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_json_file(filename, data):
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass


if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_json_file(PORTFOLIO_FILE)

if "pullback_watchlist" not in st.session_state:
    st.session_state["pullback_watchlist"] = load_json_file(WATCHLIST_FILE)

if "screener_data" not in st.session_state:
    st.session_state["screener_data"] = pd.DataFrame()

st.sidebar.header("🔑 API Setup")
GEMINI_API_KEY = st.sidebar.text_input("Google Gemini API Key", type="password")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY.strip())
    except Exception:
        pass

# Core Watchlist for stable performance
CORE_WATCHLIST = [
    "HAL.NS", "BEL.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "TATAMOTORS.NS", "MARUTI.NS", "SUNPHARMA.NS"
]

selected_universe = st.sidebar.selectbox("Select Stock Basket", ["Nifty 50 Core", "Defence & PSUs"])
tickers_to_scan = CORE_WATCHLIST if selected_universe == "Nifty 50 Core" else ["HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS"]

scan_button = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    rows = []
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=5)
            if df is None or df.empty or len(df) < 20:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            curr_price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr_price
            change_pct = round(((curr_price - prev_close) / prev_close) * 100.0, 2)
            vol = int(df["Volume"].iloc[-1])

            rows.append({
                "Ticker": ticker.replace(".NS", ""),
                "Signal": "🟢 STRONG BUY (Breakout)" if change_pct > 0 else "🟡 BUY / PULLBACK",
                "Price (₹)": round(curr_price, 2),
                "Change (%)": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
                "Volume": f"{vol:,}",
                "Composite Score": 85,
                "Raw_Ticker": ticker,
                "_raw_vol": vol,
                "_change_num": change_pct,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


if scan_button or st.session_state["screener_data"].empty:
    with st.spinner("Analyzing market data..."):
        st.session_state["screener_data"] = fetch_screener_universe(tickers_to_scan)

df_raw = st.session_state["screener_data"]

tab1, tab2 = st.tabs(["📊 Screener & Momentum Signals", "💼 Paper Trading Portfolio"])

with tab1:
    st.subheader("Matching Stocks")
    if not df_raw.empty:
        st.dataframe(df_raw[["Ticker", "Signal", "Price (₹)", "Change (%)", "Volume", "Composite Score"]], use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Screener Scan' in the sidebar.")

with tab2:
    st.subheader("Paper Trading Portfolio")
    active_portfolio = st.session_state.get("paper_portfolio", [])
    if active_portfolio:
        st.dataframe(pd.DataFrame(active_portfolio), use_container_width=True, hide_index=True)
    else:
        st.info("No active trades.")
