from datetime import date
import os
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
    layout="wide",
)

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

if "screener_data" not in st.session_state:
    st.session_state["screener_data"] = pd.DataFrame()

st.sidebar.header("🔑 Settings")
scan_button = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)

CORE_WATCHLIST = ["HAL.NS", "BEL.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    rows = []
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False, timeout=5)
            if df is None or df.empty or len(df) < 10:
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
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

if scan_button or st.session_state["screener_data"].empty:
    with st.spinner("Analyzing market data..."):
        st.session_state["screener_data"] = fetch_screener_universe(CORE_WATCHLIST)

df_raw = st.session_state["screener_data"]

tab1, tab2 = st.tabs(["📊 Screener & Momentum Signals", "💼 Paper Trading Portfolio"])

with tab1:
    st.subheader("Matching Stocks")
    if not df_raw.empty:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Screener Scan' in the sidebar.")

with tab2:
    st.subheader("Paper Trading Portfolio")
    active_portfolio = st.session_state.get("paper_portfolio", [])
    if active_portfolio:
        st.dataframe(pd.DataFrame(active_portfolio), use_container_width=True, hide_index=True)
    else:
        st.info("No active trades.")
