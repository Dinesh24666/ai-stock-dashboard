import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Config
st.set_page_config(
    page_title="Indian Market Screener", page_icon="🔍", layout="wide"
)
st.title("🔍 Indian Market AI Screener & Dashboard")

# 2. Sidebar - Setup & Preset Universes
st.sidebar.header("⚙️ Configuration")
GEMINI_API_KEY = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get a free key from Google AI Studio",
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())

# Universe Selection
UNIVERSE_PRESETS = {
    "Nifty Bluechips (Top 10)": [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "BHARTIARTL.NS",
        "INFY.NS",
        "LT.NS",
        "SBIN.NS",
        "ITC.NS",
        "TATAMOTORS.NS",
    ],
    "Banking & NBFCs": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "BAJFINANCE.NS",
        "LTF.NS",
    ],
    "IT & Tech": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "LTIM.NS",
    ],
    "Custom Watchlist": [],
}

selected_universe = st.sidebar.selectbox(
    "Select Stock Basket", list(UNIVERSE_PRESETS.keys())
)

if selected_universe == "Custom Watchlist":
    custom_input = st.sidebar.text_area(
        "Enter Tickers (comma-separated)", "RELIANCE, ICICIBANK, LTF, TCS"
    )
    tickers_to_scan = [
        f"{t.strip().upper()}.NS"
        if not (
            t.strip().upper().endswith(".NS")
            or t.strip().upper().endswith(".BO")
        )
        else t.strip().upper()
        for t in custom_input.split(",")
        if t.strip()
    ]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

# 3. Screener Sliders & Filters (Screener.in style)
st.sidebar.header("🎯 Quantitative Filters")
min_roe = st.sidebar.slider("Min Return on Equity (ROE %)", 0, 40, 12)
max_pe = st.sidebar.slider("Max P/E Ratio", 5, 100, 45)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 3.0, 1.5, step=0.1)
only_bullish_sma = st.sidebar.checkbox(
    "Only Bullish Trend (Price > 50 & 200 SMA)", value=False
)


# 4. Cached Batch Fetcher & Calculator
@st.cache_data(ttl=1800)
def fetch_screener_universe(ticker_list):
    rows = []
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist.empty or len(hist) < 20:
                continue

            info = t.info
            curr_price = hist["Close"].iloc[-1]
            sma_50 = (
                hist["Close"].rolling(50).mean().iloc[-1]
                if len(hist) >= 50
                else curr_price
            )
            sma_200 = (
                hist["Close"].rolling(200).mean().iloc[-1]
                if len(hist) >= 200
                else curr_price
            )

            roe = (info.get("returnOnEquity") or 0.10) * 100
            pe = info.get("trailingPE") or info.get("forwardPE") or 25.0
            de = (info.get("debtToEquity") or 50.0) / 100.0
            mcap_cr = (info.get("marketCap") or 0) / 1e7  # Converted to ₹ Crores
            fcf = info.get("freeCashflow") or 0
            rev = info.get("totalRevenue") or 1
            fcf_margin = (fcf / rev) * 100.0 if rev > 0 else 0.0

            # Composite Score (0-100)
            fund_score = min(
                100,
                max(0, (roe / 25.0) * 60 + ((2.0 - min(de, 2.0)) / 2.0) * 40),
            )
            tech_score = (
                (35 if curr_price >= sma_50 else 0)
                + (35 if curr_price >= sma_200 else 0)
                + (30 if sma_50 >= sma_200 else 0)
            )
            composite = round(0.6 * fund_score + 0.4 * tech_score, 1)

            rows.append(
                {
                    "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                    "Company": info.get(
                        "shortName", ticker.replace(".NS", "")
                    ),
                    "Price (₹)": round(curr_price, 2),
                    "P/E": round(pe, 1),
                    "ROE (%)": round(roe, 1),
                    "D/E": round(de, 2),
                    "FCF Margin (%)": round(fcf_margin, 1),
                    "Market Cap (₹ Cr)": round(mcap_cr, 0),
                    "SMA 50": round(sma_50, 2),
                    "SMA 200": round(sma_200, 2),
                    "Above 200 SMA": curr_price >= sma_200,
                    "Composite Score": composite,
                    "Raw_Ticker": ticker,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


# Fetch Data
with st.spinner("Screening market data..."):
    df_raw = fetch_screener_universe(tickers_to_scan)

# 5. Apply User Filters
if not df_raw.empty:
    filtered_df = df_raw[
        (df_raw["ROE (%)"] >= min_roe)
        & (df_raw["P/E"] <= max_pe)
        & (df_raw["D/E"] <= max_de)
    ]
    if only_bullish_sma:
        filtered_df = filtered_df[
            (filtered_df["Price (₹)"] >= filtered_df["SMA_50"])
            & (filtered_df["Price (₹)"] >= filtered_df["SMA_200"])
        ]

    # Tabs: Screener View vs Deep-Dive Analysis
    tab_screener, tab_deepdive = st.tabs(
        ["📊 Screener Results", "🔬 Single Stock Deep-Dive & AI Thesis"]
    )

    # --- TAB 1: SCREENER TABLE ---
    with tab_screener:
        st.subheader(
            f"Matching Stocks ({len(filtered_df)} of {len(df_raw)} passed filters)"
        )

        display_cols = [
            "Ticker",
            "Company",
            "Price (₹)",
            "Composite Score",
            "P/E",
            "ROE (%)",
            "D/E",
            "FCF Margin (%)",
            "Market Cap (₹ Cr)",
            "Above 200 SMA",
        ]
        st.dataframe(
            filtered_df[display_cols].sort_values(
                by="Composite Score", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- TAB 2: DEEP DIVE ---
    with tab_deepdive:
        selected_stock = st.selectbox(
            "Select stock from results for deep dive:",
            filtered_df["Raw_Ticker"].tolist()
            if not filtered_df.empty
            else df_raw["Raw_Ticker"].tolist(),
        )

        if selected_stock:
            t = yf.Ticker(selected_stock)
            hist = t.history(period="1y")
            info = t.info

            hist["SMA_50"] = hist["Close"].rolling(50).mean()
            hist["SMA_200"] = hist["Close"].rolling(200).mean()

            # Candlestick Chart
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
                        line=dict(color="royalblue", width=1.5),
                        name="200 SMA",
                    ),
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # AI Investment Thesis
            st.subheader("🤖 AI Investment Thesis")
            if st.button("Generate AI Thesis for " + selected_stock):
                if not GEMINI_API_KEY:
                    st.warning("Please provide your Gemini API Key in sidebar.")
                else:
                    stock_row = df_raw[
                        df_raw["Raw_Ticker"] == selected_stock
                    ].iloc[0]
                    prompt = f"""
                    Analyze this Indian stock:
                    - Company: {stock_row['Company']} ({stock_row['Ticker']})
                    - Price: ₹{stock_row['Price (₹)']}
                    - P/E: {stock_row['P/E']} | ROE: {stock_row['ROE (%)']}% | Debt/Equity: {stock_row['D/E']}
                    - FCF Margin: {stock_row['FCF Margin (%)']}%
                    - Quality Composite Score: {stock_row['Composite Score']}/100

                    Provide a Screener.in-style analyst breakdown:
                    1. **Pros / Moat Factors**
                    2. **Cons / Red Flags**
                    3. **Investment Verdict** (Bullish / Neutral / Bearish)
                    """
                    with st.spinner("Generating AI Analysis..."):
                        try:
                            model_candidates = [
                                "gemini-1.5-flash",
                                "gemini-1.5-flash-latest",
                                "gemini-2.0-flash",
                                "gemini-pro",
                            ]
                            res_text = None
                            for mod in model_candidates:
                                try:
                                    m = genai.GenerativeModel(mod)
                                    res = m.generate_content(prompt)
                                    if res and res.text:
                                        res_text = res.text
                                        break
                                except Exception:
                                    continue
                            if res_text:
                                st.markdown(res_text)
                            else:
                                st.error("Could not generate summary.")
                        except Exception as e:
                            st.error(f"Error: {e}")
else:
    st.warning(
        "No data retrieved. Check your internet connection or selected universe."
    )
