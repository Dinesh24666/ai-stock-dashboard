import os
import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Config
st.set_page_config(
    page_title="Indian Market AI Screener",
    page_icon="📈",
    layout="wide",
)

st.title("⚡ Indian Market AI Stock Screener & Dashboard")

# 2. Sidebar - Gemini Key
st.sidebar.header("🔑 API Setup")
GEMINI_API_KEY = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get a free key from Google AI Studio (aistudio.google.com)",
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())


# Helper to fetch All NSE Stocks
@st.cache_data(ttl=86400)
def get_all_nse_symbols():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_nse = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
        symbols = [f"{sym}.NS" for sym in df_nse["SYMBOL"].dropna().unique()]
        return symbols
    except Exception:
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
            "INFY.NS", "LT.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "TATAMOTORS.NS",
            "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
            "AXISBANK.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "TITAN.NS",
            "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS",
            "COALINDIA.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", "ASIANPAINT.NS", "HAL.NS",
            "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS", "IRFC.NS", "LTF.NS", "ZOMATO.NS"
        ]


# Sector Baskets
UNIVERSE_PRESETS = {
    "Nifty 50 Core": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
        "INFY.NS", "LT.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS",
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
        "TITAN.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "ULTRACEMCO.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", "ASIANPAINT.NS"
    ],
    "All NSE Stocks (Full Market)": "ALL_NSE",
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "LTF.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS",
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS"
    ],
    "IT & Technology": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "KPITTECH.NS"
    ],
    "Automobile & EV": [
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "TVSMOTOR.NS", "EICHERMOT.NS", "BHARATFORG.NS", "SONACOMS.NS", "MOTHERSON.NS"
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "MANKIND.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS", "MAXHEALTH.NS"
    ],
    "FMCG & Retail": [
        "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "VBL.NS", "MARICO.NS", "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "DMART.NS"
    ],
    "Energy, Oil & Power": [
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "BPCL.NS", "IOC.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "NHPC.NS", "IREDA.NS"
    ],
    "Defence, Rail & PSUs": [
        "HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS",
        "IRFC.NS", "COCHINSHIP.NS", "BDL.NS", "CONCOR.NS"
    ],
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
        "NMDC.NS", "SAIL.NS", "NATIONALUM.NS"
    ],
    "Custom Watchlist": [],
}

st.sidebar.header("🎯 Universe Selection")
selected_universe = st.sidebar.selectbox("Select Stock Basket", list(UNIVERSE_PRESETS.keys()))

if selected_universe == "All NSE Stocks (Full Market)":
    all_symbols = get_all_nse_symbols()
    scan_limit = st.sidebar.slider("Number of NSE Stocks to Scan", 10, min(500, len(all_symbols)), 50, step=10)
    tickers_to_scan = all_symbols[:scan_limit]
elif selected_universe == "Custom Watchlist":
    custom_input = st.sidebar.text_area(
        "Enter Tickers (comma-separated)",
        "RELIANCE, ICICIBANK, LTF, TCS, TATAMOTORS, HAL",
    )
    raw_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    tickers_to_scan = [
        t if (t.endswith(".NS") or t.endswith(".BO")) else f"{t}.NS"
        for t in raw_tickers
    ]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

# 3. Sidebar Filters
# --- Composite Score Filter ---
st.sidebar.header("⭐ Composite Score Filter")
score_filter_type = st.sidebar.selectbox(
    "Score Condition",
    ["No Score Filter", "Greater than or equal (>=)", "Less than or equal (<=)"]
)
if score_filter_type != "No Score Filter":
    target_score = st.sidebar.slider("Target Composite Score", 0, 100, 70)
else:
    target_score = None

# --- Fundamental Filters ---
st.sidebar.header("📊 Fundamental Filters")
min_roe = st.sidebar.slider("Min Return on Equity (ROE %)", 0, 40, 10)
max_pe = st.sidebar.slider("Max P/E Ratio", 5, 120, 60)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 3.5, 2.0, step=0.1)

# --- Technical Filters ---
st.sidebar.header("📈 Technical Filters")
rsi_range = st.sidebar.slider("RSI (14) Range", 0, 100, (20, 80))
max_dist_52w_high = st.sidebar.slider("Within % of 52-Week High", 0, 50, 30, help="e.g. 20% means stock is at most 20% below its 52W high")

sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    ["Any Trend", "Price > 50 SMA", "Price > 200 SMA", "Price > Both 50 & 200 SMA", "Golden Cross (50 SMA > 200 SMA)"]
)
only_volume_surge = st.sidebar.checkbox("Volume Surge (Today > 20-Day Avg Volume)", value=False)


# 4. Helper RSI Calculation
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi) if not pd.isna(rsi) else 50.0


# 5. Cached Batch Fetcher
@st.cache_data(ttl=1800)
def fetch_screener_universe(ticker_list):
    rows = []
    progress_bar = st.progress(0, text="Fetching stock metrics...")
    total = len(ticker_list)

    for idx, ticker in enumerate(ticker_list):
        progress_bar.progress((idx + 1) / total, text=f"Scanning {ticker} ({idx+1}/{total})...")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist is None or hist.empty or len(hist) < 20:
                continue

            info = t.info
            curr_price = hist["Close"].iloc[-1]
            sma_50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else curr_price
            sma_200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else curr_price
            high_52w = hist["High"].max()
            dist_52w_high = ((high_52w - curr_price) / high_52w) * 100.0

            # Technicals
            rsi_val = compute_rsi(hist["Close"], 14)
            avg_vol_20 = hist["Volume"].rolling(20).mean().iloc[-1] if len(hist) >= 20 else hist["Volume"].iloc[-1]
            curr_vol = hist["Volume"].iloc[-1]
            vol_surge = bool(curr_vol > avg_vol_20)

            # Fundamentals
            roe = (info.get("returnOnEquity") or 0.10) * 100.0
            pe = info.get("trailingPE") or info.get("forwardPE") or 25.0
            de = (info.get("debtToEquity") or 50.0) / 100.0
            mcap_cr = (info.get("marketCap") or 0) / 1e7
            fcf = info.get("freeCashflow") or 0
            rev = info.get("totalRevenue") or 1
            fcf_margin = (fcf / rev) * 100.0 if rev > 0 else 0.0

            # Composite Score (0-100)
            fund_score = min(100, max(0, (roe / 25.0) * 60 + ((2.0 - min(de, 2.0)) / 2.0) * 40))
            rsi_score = max(0, 100 - 4 * abs(rsi_val - 60))
            trend_score = (
                (30 if curr_price >= sma_50 else 0)
                + (40 if curr_price >= sma_200 else 0)
                + (30 if sma_50 >= sma_200 else 0)
            )
            tech_score = 0.5 * trend_score + 0.5 * rsi_score
            composite = round(0.55 * fund_score + 0.45 * tech_score, 1)

            rows.append({
                "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "Company": info.get("shortName") or info.get("longName") or ticker.replace(".NS", ""),
                "Price (₹)": round(curr_price, 2),
                "Composite Score": composite,
                "RSI (14)": round(rsi_val, 1),
                "P/E": round(pe, 1),
                "ROE (%)": round(roe, 1),
                "D/E": round(de, 2),
                "52W High (₹)": round(high_52w, 2),
                "From 52W High (%)": round(dist_52w_high, 1),
                "FCF Margin (%)": round(fcf_margin, 1),
                "Market Cap (₹ Cr)": round(mcap_cr, 0),
                "SMA_50": round(sma_50, 2),
                "SMA_200": round(sma_200, 2),
                "Vol Surge": vol_surge,
                "Raw_Ticker": ticker,
            })
        except Exception:
            continue

    progress_bar.empty()
    return pd.DataFrame(rows)


# Fetch Data
if tickers_to_scan:
    df_raw = fetch_screener_universe(tickers_to_scan)
else:
    df_raw = pd.DataFrame()

# 6. Apply User Filters
if not df_raw.empty:
    filtered_df = df_raw.copy()

    # Composite Score Filter
    if score_filter_type == "Greater than or equal (>=)":
        filtered_df = filtered_df[filtered_df["Composite Score"] >= target_score]
    elif score_filter_type == "Less than or equal (<=)":
        filtered_df = filtered_df[filtered_df["Composite Score"] <= target_score]

    # Fundamentals Filters
    filtered_df = filtered_df[
        (filtered_df["ROE (%)"] >= min_roe)
        & (filtered_df["P/E"] <= max_pe)
        & (filtered_df["D/E"] <= max_de)
    ]

    # Technical Filters
    filtered_df = filtered_df[
        (filtered_df["RSI (14)"] >= rsi_range[0])
        & (filtered_df["RSI (14)"] <= rsi_range[1])
        & (filtered_df["From 52W High (%)"] <= max_dist_52w_high)
    ]

    # Moving Average Alignment Filter
    if sma_trend_filter == "Price > 50 SMA":
        filtered_df = filtered_df[filtered_df["Price (₹)"] >= filtered_df["SMA_50"]]
    elif sma_trend_filter == "Price > 200 SMA":
        filtered_df = filtered_df[filtered_df["Price (₹)"] >= filtered_df["SMA_200"]]
    elif sma_trend_filter == "Price > Both 50 & 200 SMA":
        filtered_df = filtered_df[
            (filtered_df["Price (₹)"] >= filtered_df["SMA_50"])
            & (filtered_df["Price (₹)"] >= filtered_df["SMA_200"])
        ]
    elif sma_trend_filter == "Golden Cross (50 SMA > 200 SMA)":
        filtered_df = filtered_df[filtered_df["SMA_50"] >= filtered_df["SMA_200"]]

    if only_volume_surge:
        filtered_df = filtered_df[filtered_df["Vol Surge"] == True]

    # Tabs: Screener View vs Deep-Dive Analysis
    tab_screener, tab_deepdive = st.tabs(
        ["📊 Screener Results", "🔬 Single Stock Deep-Dive & AI Thesis"]
    )

    # --- TAB 1: SCREENER TABLE ---
    with tab_screener:
        st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)} passed filters)")

        display_cols = [
            "Ticker",
            "Company",
            "Price (₹)",
            "Composite Score",
            "RSI (14)",
            "From 52W High (%)",
            "P/E",
            "ROE (%)",
            "D/E",
            "Vol Surge",
            "Market Cap (₹ Cr)",
        ]
        st.dataframe(
            filtered_df[display_cols].sort_values(by="Composite Score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # --- TAB 2: DEEP DIVE ---
    with tab_deepdive:
        stock_options = (
            filtered_df["Raw_Ticker"].tolist()
            if not filtered_df.empty
            else df_raw["Raw_Ticker"].tolist()
        )
        selected_stock = st.selectbox("Select stock for deep dive & technical chart:", stock_options)

        if selected_stock:
            t = yf.Ticker(selected_stock)
            hist = t.history(period="1y")
            info = t.info

            if not hist.empty:
                hist["SMA_50"] = hist["Close"].rolling(50).mean()
                hist["SMA_200"] = hist["Close"].rolling(200).mean()

                # Candlestick Chart
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name="Price",
                    ),
                    go.Scatter(x=hist.index, y=hist["SMA_50"], line=dict(color="orange", width=1.5), name="50 SMA"),
                    go.Scatter(x=hist.index, y=hist["SMA_200"], line=dict(color="royalblue", width=1.5), name="200 SMA"),
                ])
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
                        st.warning("Please provide your Gemini API Key in the left sidebar.")
                    else:
                        stock_match = df_raw[df_raw["Raw_Ticker"] == selected_stock]
                        stock_row = stock_match.iloc[0] if not stock_match.empty else None

                        prompt = f"""
                        Analyze this Indian stock:
                        - Company: {info.get('longName', selected_stock)} ({selected_stock})
                        - Sector: {info.get('sector', 'N/A')}
                        - Current Price: ₹{hist['Close'].iloc[-1]:.2f}
                        - Technicals: RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}, From 52W High: {stock_row['From 52W High (%)'] if stock_row is not None else 'N/A'}%
                        - Fundamentals: P/E: {stock_row['P/E'] if stock_row is not None else 'N/A'} | ROE: {stock_row['ROE (%)'] if stock_row is not None else 'N/A'}% | Debt/Equity: {stock_row['D/E'] if stock_row is not None else 'N/A'}
                        - Quality Composite Score: {stock_row['Composite Score'] if stock_row is not None else 'N/A'}/100

                        Provide a Screener.in-style analyst breakdown:
                        1. **Pros & Technical Moat**
                        2. **Key Risks & Valuation Flags**
                        3. **Investment Verdict** (Bullish / Neutral / Bearish with target reasoning)
                        """
                        with st.spinner("Generating AI Analysis with Gemini 3.6..."):
                            try:
                                model = genai.GenerativeModel("gemini-3.6-flash")
                                res = model.generate_content(prompt)
                                st.markdown(res.text)
                            except Exception as err:
                                st.error(f"Error generating AI thesis: {err}")
else:
    st.warning("No data retrieved. Verify your stock basket selection or network connection.")
