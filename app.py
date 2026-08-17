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

# 2. Sidebar - API Key
st.sidebar.header("🔑 API Setup")
GEMINI_API_KEY = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get a free key from Google AI Studio (aistudio.google.com)",
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())

# Universe Presets
UNIVERSE_PRESETS = {
    "Nifty 50 Core": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
        "INFY.NS", "LT.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS",
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
        "TITAN.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "ULTRACEMCO.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "NESTLEIND.NS", "ASIANPAINT.NS"
    ],
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
    "All NSE Stocks (Full Market)": "ALL_NSE",
    "Custom Watchlist": [],
}

@st.cache_data(ttl=86400)
def get_all_nse_symbols():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_nse = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
        return [f"{sym}.NS" for sym in df_nse["SYMBOL"].dropna().unique()]
    except Exception:
        return UNIVERSE_PRESETS["Nifty 50 Core"]

# Sidebar Universe Selection
st.sidebar.header("🎯 Universe Selection")
selected_universe = st.sidebar.selectbox("Select Stock Basket", list(UNIVERSE_PRESETS.keys()))

if selected_universe == "All NSE Stocks (Full Market)":
    all_symbols = get_all_nse_symbols()
    scan_limit = st.sidebar.slider("Number of NSE Stocks to Scan", 10, min(200, len(all_symbols)), 40, step=10)
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

# 3. Quantitative & Technical Filters
st.sidebar.header("⭐ Composite Score Filter")
score_filter_type = st.sidebar.selectbox(
    "Score Condition",
    ["No Score Filter", "Greater than or equal (>=)", "Less than or equal (<=)"]
)
target_score = st.sidebar.slider("Target Composite Score", 0, 100, 70) if score_filter_type != "No Score Filter" else None

st.sidebar.header("📊 Fundamental Filters")
min_roe = st.sidebar.slider("Min Return on Equity (ROE %)", -20, 50, 0)
max_pe = st.sidebar.slider("Max P/E Ratio", 5, 200, 100)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 10.0, 5.0, step=0.1)

st.sidebar.header("📈 Technical Filters")
rsi_range = st.sidebar.slider("RSI (14) Range", 0, 100, (10, 90))
max_dist_52w_high = st.sidebar.slider("Within % of 52-Week High", 0, 100, 50)
sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    ["Any Trend", "Price > 50 SMA", "Price > 200 SMA", "Price > Both 50 & 200 SMA", "Golden Cross (50 SMA > 200 SMA)"]
)
only_volume_surge = st.sidebar.checkbox("Volume Surge (Today > 20-Day Avg Volume)", value=False)


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi) if not pd.isna(rsi) else 50.0


# 4. Cached Batch Fetcher
@st.cache_data(ttl=3600)
def fetch_screener_universe(ticker_list):
    rows = []
    progress_bar = st.progress(0, text="Fetching real-time stock data...")
    total = len(ticker_list)

    for idx, ticker in enumerate(ticker_list):
        progress_bar.progress((idx + 1) / total, text=f"Analyzing {ticker} ({idx+1}/{total})...")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist is None or hist.empty or len(hist) < 20:
                continue

            curr_price = float(hist["Close"].iloc[-1])
            sma_50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else curr_price
            sma_200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else curr_price
            high_52w = float(hist["High"].max())
            dist_52w_high = max(0.0, ((high_52w - curr_price) / high_52w) * 100.0)

            rsi_val = compute_rsi(hist["Close"], 14)
            avg_vol_20 = float(hist["Volume"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else float(hist["Volume"].iloc[-1])
            vol_surge = bool(float(hist["Volume"].iloc[-1]) > avg_vol_20)

            info = {}
            try:
                info = t.get_info()
            except Exception:
                info = {}

            company_name = info.get("shortName") or info.get("longName") or ticker.replace(".NS", "")
            sector = info.get("sector") or info.get("industry") or "Diversified"

            raw_mcap = None
            try:
                raw_mcap = t.fast_info.market_cap
            except Exception:
                raw_mcap = info.get("marketCap")
            mcap_cr = round((raw_mcap / 1e7), 1) if raw_mcap and raw_mcap > 0 else 0.0

            pe_val = info.get("trailingPE") or info.get("forwardPE")
            pe = round(float(pe_val), 1) if pe_val and pe_val > 0 else np.nan

            roe_val = info.get("returnOnEquity")
            roe = round(float(roe_val) * 100.0, 1) if roe_val is not None else np.nan

            de_val = info.get("debtToEquity")
            de = round(float(de_val) / 100.0, 2) if de_val is not None else np.nan

            safe_roe = roe if not np.isnan(roe) else 12.0
            safe_de = de if not np.isnan(de) else 0.8
            fund_score = min(100, max(0, (safe_roe / 25.0) * 60 + ((2.0 - min(safe_de, 2.0)) / 2.0) * 40))
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
                "Company": company_name,
                "Sector": sector,
                "Price (₹)": round(curr_price, 2),
                "Composite Score": composite,
                "RSI (14)": round(rsi_val, 1),
                "From 52W High (%)": round(dist_52w_high, 1),
                "P/E": pe if not np.isnan(pe) else np.nan,
                "ROE (%)": roe if not np.isnan(roe) else np.nan,
                "D/E": de if not np.isnan(de) else np.nan,
                "Vol Surge": vol_surge,
                "Market Cap (₹ Cr)": mcap_cr,
                "SMA_50": round(sma_50, 2),
                "SMA_200": round(sma_200, 2),
                "Raw_Ticker": ticker,
                "_pe_num": pe if not np.isnan(pe) else 999.0,
                "_roe_num": roe if not np.isnan(roe) else -999.0,
                "_de_num": de if not np.isnan(de) else 999.0,
            })
        except Exception:
            continue

    progress_bar.empty()
    return pd.DataFrame(rows)


if tickers_to_scan:
    df_raw = fetch_screener_universe(tickers_to_scan)
else:
    df_raw = pd.DataFrame()

# 5. Apply User Filters
if not df_raw.empty:
    filtered_df = df_raw.copy()

    if score_filter_type == "Greater than or equal (>=)":
        filtered_df = filtered_df[filtered_df["Composite Score"] >= target_score]
    elif score_filter_type == "Less than or equal (<=)":
        filtered_df = filtered_df[filtered_df["Composite Score"] <= target_score]

    filtered_df = filtered_df[
        (filtered_df["_roe_num"] >= min_roe)
        & (filtered_df["_pe_num"] <= max_pe)
        & (filtered_df["_de_num"] <= max_de)
    ]

    filtered_df = filtered_df[
        (filtered_df["RSI (14)"] >= rsi_range[0])
        & (filtered_df["RSI (14)"] <= rsi_range[1])
        & (filtered_df["From 52W High (%)"] <= max_dist_52w_high)
    ]

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

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = (
            filtered_df["Raw_Ticker"].iloc[0]
            if not filtered_df.empty
            else df_raw["Raw_Ticker"].iloc[0]
        )

    tab_screener, tab_deepdive = st.tabs(
        ["📊 Screener Results", "🔬 Single Stock Chart & AI Thesis"]
    )

    # --- TAB 1: SCREENER TABLE WITH IN-TAB SORTING FILTER ---
    with tab_screener:
        st.info("💡 **Tip:** Click any row in the table below to load its 9/20 EMA chart in the Single Stock tab.")
        
        # In-tab Sorting Controls Header
        col_title, col_sort_by, col_sort_dir = st.columns([2, 1.2, 1])
        with col_title:
            st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)})")
        with col_sort_by:
            sort_metric = st.selectbox(
                "Sort Results By:",
                ["Composite Score", "Price (₹)", "RSI (14)", "From 52W High (%)", "P/E", "ROE (%)", "D/E", "Market Cap (₹ Cr)"],
                index=0
            )
        with col_sort_dir:
            sort_order = st.radio(
                "Order:",
                ["High to Low (Desc)", "Low to High (Asc)"],
                horizontal=True
            )

        sort_col_map = {
            "Composite Score": "Composite Score",
            "Price (₹)": "Price (₹)",
            "RSI (14)": "RSI (14)",
            "From 52W High (%)": "From 52W High (%)",
            "P/E": "_pe_num",
            "ROE (%)": "_roe_num",
            "D/E": "_de_num",
            "Market Cap (₹ Cr)": "Market Cap (₹ Cr)"
        }
        
        target_sort_col = sort_col_map.get(sort_metric, "Composite Score")
        ascending_flag = (sort_order == "Low to High (Asc)")
        
        sorted_results_df = filtered_df.sort_values(by=target_sort_col, ascending=ascending_flag)

        display_cols = [
            "Ticker",
            "Company",
            "Sector",
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

        table_data = sorted_results_df[display_cols].copy()
        
        # Replace internal NaNs with clean "N/A" strings for clean presentation
        table_data["P/E"] = table_data["P/E"].fillna("N/A")
        table_data["ROE (%)"] = table_data["ROE (%)"].fillna("N/A")
        table_data["D/E"] = table_data["D/E"].fillna("N/A")

        selection_event = st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selection_event and selection_event.selection and selection_event.selection.rows:
            selected_row_idx = selection_event.selection.rows[0]
            clicked_ticker_sym = table_data.iloc[selected_row_idx]["Ticker"]
            st.session_state.selected_ticker = f"{clicked_ticker_sym}.NS"

    # --- TAB 2: DEEP DIVE & CHART VIEW ---
    with tab_deepdive:
        stock_options = (
            sorted_results_df["Raw_Ticker"].tolist()
            if not sorted_results_df.empty
            else df_raw["Raw_Ticker"].tolist()
        )

        default_index = (
            stock_options.index(st.session_state.selected_ticker)
            if st.session_state.selected_ticker in stock_options
            else 0
        )
        selected_stock = st.selectbox(
            "Selected Stock:", stock_options, index=default_index
        )
        st.session_state.selected_ticker = selected_stock

        if selected_stock:
            t = yf.Ticker(selected_stock)
            hist = t.history(period="1y")
            try:
                info = t.get_info()
            except Exception:
                info = {}

            if not hist.empty:
                hist["EMA_9"] = hist["Close"].ewm(span=9, adjust=False).mean()
                hist["EMA_20"] = hist["Close"].ewm(span=20, adjust=False).mean()
                hist["SMA_50"] = hist["Close"].rolling(50).mean()
                hist["SMA_200"] = hist["Close"].rolling(200).mean()

                curr_p = float(hist["Close"].iloc[-1])
                ema9_val = float(hist["EMA_9"].iloc[-1])
                ema20_val = float(hist["EMA_20"].iloc[-1])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{curr_p:,.2f}")
                c2.metric("9 EMA", f"₹{ema9_val:,.2f}")
                c3.metric("20 EMA", f"₹{ema20_val:,.2f}")
                c4.metric(
                    "Short-Term Momentum",
                    "🚀 Bullish (9 > 20 EMA)" if ema9_val >= ema20_val else "🔻 Bearish (9 < 20 EMA)",
                )

                fig = go.Figure(data=[
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name="Price",
                    ),
                    go.Scatter(x=hist.index, y=hist["EMA_9"], line=dict(color="#00f2ff", width=1.5), name="9 EMA (Fast)"),
                    go.Scatter(x=hist.index, y=hist["EMA_20"], line=dict(color="#ffd700", width=1.5), name="20 EMA (Momentum)"),
                    go.Scatter(x=hist.index, y=hist["SMA_50"], line=dict(color="#ff9900", width=1.5), name="50 SMA"),
                    go.Scatter(x=hist.index, y=hist["SMA_200"], line=dict(color="#4d79ff", width=1.5), name="200 SMA"),
                ])
                fig.update_layout(
                    template="plotly_dark",
                    height=480,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

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
                        - Sector: {stock_row['Sector'] if stock_row is not None else info.get('sector', 'N/A')}
                        - Current Price: ₹{curr_p:.2f}
                        - Moving Averages: 9 EMA = ₹{ema9_val:.2f}, 20 EMA = ₹{ema20_val:.2f}, Short-term setup: {'Bullish Cross' if ema9_val >= ema20_val else 'Bearish'}
                        - Technicals: RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}, From 52W High: {stock_row['From 52W High (%)'] if stock_row is not None else 'N/A'}%
                        - Fundamentals: P/E: {stock_row['P/E'] if stock_row is not None else 'N/A'} | ROE: {stock_row['ROE (%)'] if stock_row is not None else 'N/A'}% | Debt/Equity: {stock_row['D/E'] if stock_row is not None else 'N/A'}
                        - Market Cap: ₹{stock_row['Market Cap (₹ Cr)'] if stock_row is not None else 'N/A'} Cr
                        - Quality Composite Score: {stock_row['Composite Score'] if stock_row is not None else 'N/A'}/100

                        Provide a clean analyst breakdown:
                        1. **Technical & EMA Trend Analysis**
                        2. **Fundamental Quality & Moat**
                        3. **Key Risks & Valuation Check**
                        4. **Actionable Verdict** (Bullish / Neutral / Bearish)
                        """
                        with st.spinner("Generating AI Analysis with Gemini 3.6..."):
                            try:
                                model = genai.GenerativeModel("gemini-3.6-flash")
                                res = model.generate_content(prompt)
                                st.markdown(res.text)
                            except Exception as err:
                                st.error(f"Error generating AI thesis: {err}")
else:
    st.warning("No stocks passed the selected filter criteria. Try relaxing the filters in the sidebar.")
