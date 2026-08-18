from datetime import date
import json
import os
import time
import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Config
st.set_page_config(
    page_title="Indian Market AI Screener & Paper Trader",
    page_icon="📈",
    layout="wide",
)

st.title("⚡ Indian Market AI Stock Screener & Paper Trading")

# --- PERSISTENT STORAGE HELPERS ---
PORTFOLIO_FILE = "portfolio.json"


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_portfolio(portfolio_data):
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio_data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")


# Initialize persistent state
if "paper_portfolio" not in st.session_state:
    st.session_state.paper_portfolio = load_portfolio()

if "ai_analysis_cache" not in st.session_state:
    st.session_state.ai_analysis_cache = {}

# 2. Sidebar - API Key (Auto-detects from secrets or text input)
st.sidebar.header("🔑 API Setup")
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")

if api_key_from_secrets:
    GEMINI_API_KEY = api_key_from_secrets.strip()
    st.sidebar.success("✅ Gemini API Key connected")
else:
    GEMINI_API_KEY = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get a free key from Google AI Studio (aistudio.google.com)",
    )

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())

# Universe Presets
NIFTY_50 = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "INFY.NS",
    "LT.NS",
    "SBIN.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "TATAMOTORS.NS",
    "M&M.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "BAJFINANCE.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "TITAN.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "ULTRACEMCO.NS",
    "COALINDIA.NS",
    "BAJAJ-AUTO.NS",
    "NESTLEIND.NS",
    "ASIANPAINT.NS",
    "TECHM.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "LTIM.NS",
    "GRASIM.NS",
    "HEROMOTOCO.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "APOLLOHOSP.NS",
    "EICHERMOT.NS",
    "DIVISLAB.NS",
    "TATACONSUM.NS",
    "BRITANNIA.NS",
    "BPCL.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS",
    "TRENT.NS",
    "BEL.NS",
]

UNIVERSE_PRESETS = {
    "Nifty 50 Core": NIFTY_50,
    "Nifty 500 (Broad Market)": "NIFTY_500",
    "Banking & Financial Services": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "LTF.NS",
        "CHOLAFIN.NS",
        "SHRIRAMFIN.NS",
        "FEDERALBNK.NS",
        "IDFCFIRSTB.NS",
        "PNB.NS",
        "BANKBARODA.NS",
    ],
    "IT & Technology": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "LTIM.NS",
        "PERSISTENT.NS",
        "COFORGE.NS",
        "MPHASIS.NS",
        "KPITTECH.NS",
    ],
    "Automobile & EV": [
        "TATAMOTORS.NS",
        "M&M.NS",
        "MARUTI.NS",
        "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS",
        "TVSMOTOR.NS",
        "EICHERMOT.NS",
        "BHARATFORG.NS",
        "SONACOMS.NS",
        "MOTHERSON.NS",
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA.NS",
        "DRREDDY.NS",
        "CIPLA.NS",
        "DIVISLAB.NS",
        "APOLLOHOSP.NS",
        "MANKIND.NS",
        "LUPIN.NS",
        "ZYDUSLIFE.NS",
        "TORNTPHARM.NS",
        "MAXHEALTH.NS",
    ],
    "Defence, Rail & PSUs": [
        "HAL.NS",
        "BEL.NS",
        "BHEL.NS",
        "MAZDOCK.NS",
        "RVNL.NS",
        "IRFC.NS",
        "COCHINSHIP.NS",
        "BDL.NS",
        "CONCOR.NS",
    ],
    "All NSE Stocks (Full Listed)": "ALL_NSE",
    "🔍 Search Specific Stocks": "SEARCH_BAR",
}


@st.cache_data(ttl=86400)
def get_nse_symbols(universe_type):
    try:
        if universe_type == "NIFTY_500":
            url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
            df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
            return [f"{sym}.NS" for sym in df["Symbol"].dropna().unique()]
        else:
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
            return [f"{sym}.NS" for sym in df["SYMBOL"].dropna().unique()]
    except Exception:
        return NIFTY_50


# Sidebar Universe Selection
st.sidebar.header("🎯 Universe Selection")
selected_universe = st.sidebar.selectbox(
    "Select Stock Basket", list(UNIVERSE_PRESETS.keys())
)

if selected_universe == "🔍 Search Specific Stocks":
    search_query = st.sidebar.text_input(
        "Search Stock Symbols (e.g. RELIANCE, TCS, LTF)", "RELIANCE, LTF, ICICIBANK, TATAMOTORS, HAL"
    )
    raw_tickers = [t.strip().upper() for t in search_query.split(",") if t.strip()]
    tickers_to_scan = [
        t if (t.endswith(".NS") or t.endswith(".BO")) else f"{t}.NS"
        for t in raw_tickers
    ]
elif selected_universe in [
    "Nifty 500 (Broad Market)",
    "All NSE Stocks (Full Listed)",
]:
    preset_type = (
        "NIFTY_500"
        if selected_universe == "Nifty 500 (Broad Market)"
        else "ALL_NSE"
    )
    all_symbols = get_nse_symbols(preset_type)
    scan_limit = st.sidebar.slider(
        "Number of Stocks to Scan",
        10,
        min(500, len(all_symbols)),
        50,
        step=10,
    )
    tickers_to_scan = all_symbols[:scan_limit]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

# 3. Quantitative & Technical Filters
st.sidebar.header("📊 Fundamental Filters")
apply_fund_filter = st.sidebar.checkbox(
    "Enable Strict Fundamental Filters", value=False
)
roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, (10, 60))
mcap_range_cr = st.sidebar.slider(
    "Market Cap Range (₹ Cr)",
    0,
    2000000,
    (100, 2000000),
    step=500,
    help="Filter by minimum and maximum market capitalization in ₹ Crores",
)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, 2.5, step=0.1)

st.sidebar.header("📈 Technical Filters")
rsi_range = st.sidebar.slider("RSI (14) Range", 0, 100, (20, 85))
max_dist_52w_high = st.sidebar.slider("Within % of 52-Week High", 0, 100, 60)
sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    [
        "Any Trend",
        "Price > 50 SMA",
        "Price > 200 SMA",
        "Price > Both 50 & 200 SMA",
        "Golden Cross (50 SMA > 200 SMA)",
    ],
)
only_volume_surge = st.sidebar.checkbox(
    "Volume Surge (Today > 20-Day Avg Volume)", value=False
)


# Wilder's Exponential Smoothing RSI
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi) if not pd.isna(rsi) else 50.0


# 4. Cached Batch Fetcher
@st.cache_data(ttl=3600)
def fetch_screener_universe(ticker_list):
    rows = []
    progress_bar = st.progress(0, text="Fetching stock metrics...")
    total = len(ticker_list)

    for idx, ticker in enumerate(ticker_list):
        progress_bar.progress(
            (idx + 1) / total, text=f"Analyzing {ticker} ({idx+1}/{total})..."
        )
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist is None or hist.empty or len(hist) < 20:
                continue

            curr_price = float(hist["Close"].iloc[-1])
            sma_50 = (
                float(hist["Close"].rolling(50).mean().iloc[-1])
                if len(hist) >= 50
                else curr_price
            )
            sma_200 = (
                float(hist["Close"].rolling(200).mean().iloc[-1])
                if len(hist) >= 200
                else curr_price
            )
            high_52w = float(hist["High"].max())
            dist_52w_high = max(
                0.0, ((high_52w - curr_price) / high_52w) * 100.0
            )

            rsi_val = compute_rsi(hist["Close"], 14)
            avg_vol_20 = (
                float(hist["Volume"].rolling(20).mean().iloc[-1])
                if len(hist) >= 20
                else float(hist["Volume"].iloc[-1])
            )
            vol_surge = bool(float(hist["Volume"].iloc[-1]) > avg_vol_20)

            info = {}
            try:
                info = t.get_info()
            except Exception:
                info = {}

            company_name = (
                info.get("shortName")
                or info.get("longName")
                or ticker.replace(".NS", "")
            )
            sector = info.get("sector") or info.get("industry") or "Diversified"

            raw_mcap = None
            try:
                raw_mcap = t.fast_info.market_cap
            except Exception:
                raw_mcap = info.get("marketCap")
            mcap_cr = (
                round((raw_mcap / 1e7), 1)
                if raw_mcap and raw_mcap > 0
                else np.nan
            )

            pe_val = info.get("trailingPE") or info.get("forwardPE")
            pe = round(float(pe_val), 1) if pe_val and pe_val > 0 else np.nan

            de_val = info.get("debtToEquity")
            de = (
                round(float(de_val) / 100.0, 2)
                if de_val is not None
                else np.nan
            )

            # ROCE Calculation
            ebit = info.get("ebitda") or (
                (info.get("operatingMargins") or 0.12)
                * (info.get("totalRevenue") or 1)
            )
            tot_assets = info.get("totalAssets") or (raw_mcap or 1)
            curr_liab = info.get("currentLiabilities") or (tot_assets * 0.3)
            cap_employed = max(1.0, tot_assets - curr_liab)

            if ebit and cap_employed > 0:
                roce = round((float(ebit) / float(cap_employed)) * 100.0, 1)
                roce = min(150.0, max(-50.0, roce))
            else:
                op_margin = (info.get("operatingMargins") or 0.12) * 100.0
                roce = round(op_margin * 1.2, 1)

            safe_roce = roce if not np.isnan(roce) else 12.0
            safe_de = de if not np.isnan(de) else 0.8
            fund_score = min(
                100,
                max(
                    0,
                    (safe_roce / 25.0) * 60
                    + ((2.0 - min(safe_de, 2.0)) / 2.0) * 40,
                ),
            )
            rsi_score = max(0, 100 - 4 * abs(rsi_val - 60))
            trend_score = (
                (30 if curr_price >= sma_50 else 0)
                + (40 if curr_price >= sma_200 else 0)
                + (30 if sma_50 >= sma_200 else 0)
            )
            tech_score = 0.5 * trend_score + 0.5 * rsi_score
            composite = round(0.55 * fund_score + 0.45 * tech_score, 1)

            rows.append(
                {
                    "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                    "Company": company_name,
                    "Sector": sector,
                    "Price (₹)": round(curr_price, 2),
                    "Composite Score": composite,
                    "ROCE (%)": roce if not np.isnan(roce) else np.nan,
                    "RSI (14)": round(rsi_val, 1),
                    "From 52W High (%)": round(dist_52w_high, 1),
                    "P/E": pe if not np.isnan(pe) else np.nan,
                    "D/E": de if not np.isnan(de) else np.nan,
                    "Vol Surge": vol_surge,
                    "Market Cap (₹ Cr)": mcap_cr,
                    "SMA_50": round(sma_50, 2),
                    "SMA_200": round(sma_200, 2),
                    "Raw_Ticker": ticker,
                    "_roce_num": roce if not np.isnan(roce) else np.nan,
                    "_de_num": de if not np.isnan(de) else np.nan,
                    "_mcap_num": mcap_cr if not np.isnan(mcap_cr) else 0.0,
                }
            )
        except Exception:
            continue

    progress_bar.empty()
    return pd.DataFrame(rows)


# Cached Single Ticker History Fetcher
@st.cache_data(ttl=1800)
def get_single_stock_history(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        return hist
    except Exception:
        return pd.DataFrame()


if tickers_to_scan:
    df_raw = fetch_screener_universe(tickers_to_scan)
else:
    df_raw = pd.DataFrame()

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = (
        df_raw["Raw_Ticker"].iloc[0] if not df_raw.empty else "RELIANCE.NS"
    )

# 5. Apply User Filters
if not df_raw.empty:
    filtered_df = df_raw.copy()

    # Fundamental Range Filters
    if apply_fund_filter:
        filtered_df = filtered_df[
            (
                filtered_df["_roce_num"].isna()
                | (
                    (filtered_df["_roce_num"] >= roce_range[0])
                    & (filtered_df["_roce_num"] <= roce_range[1])
                )
            )
            & (
                filtered_df["_mcap_num"].isna()
                | (
                    (filtered_df["_mcap_num"] >= mcap_range_cr[0])
                    & (filtered_df["_mcap_num"] <= mcap_range_cr[1])
                )
            )
            & (
                filtered_df["_de_num"].isna()
                | (filtered_df["_de_num"] <= max_de)
            )
        ]

    # Technical Filters
    filtered_df = filtered_df[
        (filtered_df["RSI (14)"] >= rsi_range[0])
        & (filtered_df["RSI (14)"] <= rsi_range[1])
        & (filtered_df["From 52W High (%)"] <= max_dist_52w_high)
    ]

    # Moving Average Alignment
    if sma_trend_filter == "Price > 50 SMA":
        filtered_df = filtered_df[
            filtered_df["Price (₹)"] >= filtered_df["SMA_50"]
        ]
    elif sma_trend_filter == "Price > 200 SMA":
        filtered_df = filtered_df[
            filtered_df["Price (₹)"] >= filtered_df["SMA_200"]
        ]
    elif sma_trend_filter == "Price > Both 50 & 200 SMA":
        filtered_df = filtered_df[
            (filtered_df["Price (₹)"] >= filtered_df["SMA_50"])
            & (filtered_df["Price (₹)"] >= filtered_df["SMA_200"])
        ]
    elif sma_trend_filter == "Golden Cross (50 SMA > 200 SMA)":
        filtered_df = filtered_df[
            filtered_df["SMA_50"] >= filtered_df["SMA_200"]
        ]

    if only_volume_surge:
        filtered_df = filtered_df[filtered_df["Vol Surge"] == True]

    # Tabs
    tab_screener, tab_deepdive, tab_watchlist = st.tabs(
        [
            "📊 Screener Results",
            "🔬 Single Stock Chart & AI Thesis",
            "💼 Paper Trading Portfolio",
        ]
    )

    # --- TAB 1: SCREENER TABLE ---
    with tab_screener:
        st.info(
            "💡 **Tip:** Click any row below to view its 9/20 EMA chart or place a Paper Trade."
        )

        col_title, col_sort_by, col_sort_dir = st.columns([2, 1.2, 1])
        with col_title:
            st.subheader(
                f"Matching Stocks ({len(filtered_df)} of {len(df_raw)})"
            )
        with col_sort_by:
            sort_metric = st.selectbox(
                "Sort Results By:",
                [
                    "Composite Score",
                    "Price (₹)",
                    "ROCE (%)",
                    "RSI (14)",
                    "From 52W High (%)",
                    "P/E",
                    "D/E",
                    "Market Cap (₹ Cr)",
                ],
                index=0,
            )
        with col_sort_dir:
            sort_order = st.radio(
                "Order:",
                ["High to Low (Desc)", "Low to High (Asc)"],
                horizontal=True,
            )

        sort_col_map = {
            "Composite Score": "Composite Score",
            "Price (₹)": "Price (₹)",
            "ROCE (%)": "_roce_num",
            "RSI (14)": "RSI (14)",
            "From 52W High (%)": "From 52W High (%)",
            "P/E": "P/E",
            "D/E": "_de_num",
            "Market Cap (₹ Cr)": "_mcap_num",
        }

        target_sort_col = sort_col_map.get(sort_metric, "Composite Score")
        ascending_flag = sort_order == "Low to High (Asc)"
        sorted_results_df = filtered_df.sort_values(
            by=target_sort_col, ascending=ascending_flag, na_position="last"
        )

        display_cols = [
            "Ticker",
            "Company",
            "Sector",
            "Price (₹)",
            "Composite Score",
            "ROCE (%)",
            "RSI (14)",
            "From 52W High (%)",
            "P/E",
            "D/E",
            "Vol Surge",
            "Market Cap (₹ Cr)",
        ]

        table_data = sorted_results_df[display_cols].copy()
        table_data["P/E"] = table_data["P/E"].fillna("N/A")
        table_data["ROCE (%)"] = table_data["ROCE (%)"].fillna("N/A")
        table_data["D/E"] = table_data["D/E"].fillna("N/A")
        table_data["Market Cap (₹ Cr)"] = table_data[
            "Market Cap (₹ Cr)"
        ].fillna("N/A")

        selection_event = st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if (
            selection_event
            and selection_event.selection
            and selection_event.selection.rows
        ):
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
            hist = get_single_stock_history(selected_stock)
            stock_match = df_raw[df_raw["Raw_Ticker"] == selected_stock]
            stock_row = stock_match.iloc[0] if not stock_match.empty else None

            if hist is not None and not hist.empty:
                hist["EMA_9"] = hist["Close"].ewm(span=9, adjust=False).mean()
                hist["EMA_20"] = (
                    hist["Close"].ewm(span=20, adjust=False).mean()
                )
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
                    "Short-Term Trend",
                    "🚀 Bullish (9 > 20 EMA)"
                    if ema9_val >= ema20_val
                    else "🔻 Bearish (9 < 20 EMA)",
                )

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
                            y=hist["EMA_9"],
                            line=dict(color="#00f2ff", width=1.5),
                            name="9 EMA (Fast)",
                        ),
                        go.Scatter(
                            x=hist.index,
                            y=hist["EMA_20"],
                            line=dict(color="#ffd700", width=1.5),
                            name="20 EMA (Momentum)",
                        ),
                        go.Scatter(
                            x=hist.index,
                            y=hist["SMA_50"],
                            line=dict(color="#ff9900", width=1.5),
                            name="50 SMA",
                        ),
                        go.Scatter(
                            x=hist.index,
                            y=hist["SMA_200"],
                            line=dict(color="#4d79ff", width=1.5),
                            name="200 SMA",
                        ),
                    ]
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=480,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_rangeslider_visible=False,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🤖 AI Investment Thesis")

                cached_thesis = st.session_state.ai_analysis_cache.get(
                    selected_stock
                )
                if cached_thesis:
                    st.markdown(cached_thesis)

                if st.button("Generate AI Thesis for " + selected_stock):
                    if not GEMINI_API_KEY:
                        st.warning(
                            "Please provide your Gemini API Key in the left sidebar."
                        )
                    else:
                        prompt = f"""
                        Analyze this Indian stock:
                        - Company: {stock_row['Company'] if stock_row is not None else selected_stock} ({selected_stock})
                        - Sector: {stock_row['Sector'] if stock_row is not None else 'N/A'}
                        - Current Price: ₹{curr_p:.2f}
                        - Moving Averages: 9 EMA = ₹{ema9_val:.2f}, 20 EMA = ₹{ema20_val:.2f}, Trend: {'Bullish Cross' if ema9_val >= ema20_val else 'Bearish'}
                        - Technicals: RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}, From 52W High: {stock_row['From 52W High (%)'] if stock_row is not None else 'N/A'}%
                        - Fundamentals: ROCE: {stock_row['ROCE (%)'] if stock_row is not None else 'N/A'}% | Debt/Equity: {stock_row['D/E'] if stock_row is not None else 'N/A'} | P/E: {stock_row['P/E'] if stock_row is not None else 'N/A'}
                        - Market Cap: ₹{stock_row['Market Cap (₹ Cr)'] if stock_row is not None else 'N/A'} Cr
                        - Quality Composite Score: {stock_row['Composite Score'] if stock_row is not None else 'N/A'}/100

                        Provide a clean analyst breakdown:
                        1. **Technical & EMA Trend Analysis**
                        2. **Fundamental Quality & ROCE Evaluation**
                        3. **Key Risks & Valuation Check**
                        4. **Actionable Verdict** (Bullish / Neutral / Bearish)
                        """
                        with st.spinner("Analyzing stock with Gemini..."):
                            model_ladder = [
                                "gemini-1.5-flash",
                                "gemini-1.5-flash-8b",
                                "gemini-2.0-flash",
                                "gemini-3.6-flash",
                            ]
                            success = False

                            for mod in model_ladder:
                                try:
                                    model = genai.GenerativeModel(mod)
                                    res = model.generate_content(prompt)
                                    if res and res.text:
                                        st.session_state.ai_analysis_cache[
                                            selected_stock
                                        ] = res.text
                                        st.markdown(res.text)
                                        success = True
                                        break
                                except Exception:
                                    time.sleep(0.8)
                                    continue

                            if not success:
                                st.error(
                                    "Rate limit reached on Free Tier. Please wait 30–45 seconds before clicking again."
                                )
            else:
                st.warning(
                    f"Historical price data for {selected_stock} is temporarily unavailable."
                )

    # --- TAB 3: WATCHLIST & PAPER TRADING (WITH REMARKS & STRATEGY) ---
    with tab_watchlist:
        st.subheader("💼 Paper Trading Portfolio & Risk Manager")

        # Order Placement Form
        with st.expander(
            "➕ Execute New Paper Trade (Manual SL & Trade Remarks)",
            expanded=True,
        ):
            col_add1, col_add2, col_add3, col_add4 = st.columns(
                [1.2, 1, 1, 1]
            )

            with col_add1:
                trade_stock = st.selectbox(
                    "Stock:",
                    df_raw["Raw_Ticker"].tolist(),
                    index=df_raw["Raw_Ticker"]
                    .tolist()
                    .index(st.session_state.selected_ticker)
                    if st.session_state.selected_ticker
                    in df_raw["Raw_Ticker"].tolist()
                    else 0,
                )
            with col_add2:
                trade_date = st.date_input("Entry Date", value=date.today())

            with col_add3:
                matched_stock = df_raw[df_raw["Raw_Ticker"] == trade_stock]
                live_price = (
                    float(matched_stock["Price (₹)"].iloc[0])
                    if not matched_stock.empty
                    else 100.0
                )
                buy_price = st.number_input(
                    "Entry Price (₹)", value=live_price, min_value=0.1, step=0.5
                )

            with col_add4:
                sl_price = st.number_input(
                    "Stop Loss (SL ₹)",
                    value=0.0,
                    min_value=0.0,
                    step=0.5,
                    help="Enter custom Stop Loss level. Leave as 0.0 if no SL is needed.",
                )

            col_sub1, col_sub2, col_btn = st.columns([1, 2.5, 1])
            with col_sub1:
                quantity = st.number_input(
                    "Quantity", value=50, min_value=1, step=1
                )
            with col_sub2:
                remarks = st.text_input(
                    "Trade Remarks / Strategy (Why did you buy?)",
                    value="9/20 EMA Bullish Cross Breakout",
                    help="Record the reason or setup condition for entering this trade",
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📥 Execute Trade", use_container_width=True):
                    company_name = (
                        matched_stock["Company"].iloc[0]
                        if not matched_stock.empty
                        else trade_stock
                    )
                    new_trade = {
                        "Date": str(trade_date),
                        "Ticker": trade_stock.replace(".NS", ""),
                        "Company": company_name,
                        "Buy Price (₹)": buy_price,
                        "SL (₹)": sl_price,
                        "Qty": quantity,
                        "Remarks": remarks.strip(),
                        "Invested (₹)": round(buy_price * quantity, 2),
                        "Raw_Ticker": trade_stock,
                    }
                    st.session_state.paper_portfolio.append(new_trade)
                    save_portfolio(st.session_state.paper_portfolio)

                    sl_msg = (
                        f"(SL: ₹{sl_price})" if sl_price > 0 else "(No SL set)"
                    )
                    st.success(
                        f"Executed buy for {quantity} shares of {trade_stock.replace('.NS', '')} at ₹{buy_price} {sl_msg}!"
                    )

        # Portfolio Tracking
        if st.session_state.paper_portfolio:
            portfolio_rows = []
            open_invested = 0.0
            open_current_val = 0.0
            unrealised_pnl_total = 0.0
            realised_pnl_total = 0.0

            for pos in st.session_state.paper_portfolio:
                sym = pos.get("Raw_Ticker", "")
                m_row = df_raw[df_raw["Raw_Ticker"] == sym]
                buy_p = float(pos.get("Buy Price (₹)", 0.0))
                curr_p = (
                    float(m_row["Price (₹)"].iloc[0])
                    if not m_row.empty
                    else buy_p
                )
                invested = float(
                    pos.get("Invested (₹)", buy_p * pos.get("Qty", 1))
                )
                sl = float(pos.get("SL (₹)", 0.0))
                qty = int(pos.get("Qty", 1))
                pos_date = str(pos.get("Date", str(date.today())))
                pos_remarks = str(pos.get("Remarks", "Discretionary"))

                if sl > 0 and curr_p <= sl:
                    status = "🔴 SL Hit (Closed)"
                    exit_price = sl
                    pnl = round((exit_price - buy_p) * qty, 2)
                    pnl_pct = (
                        round((pnl / invested) * 100.0, 2)
                        if invested > 0
                        else 0.0
                    )
                    realised_pnl_total += pnl
                else:
                    status = "🟢 Open"
                    pnl = round((curr_p - buy_p) * qty, 2)
                    pnl_pct = (
                        round((pnl / invested) * 100.0, 2)
                        if invested > 0
                        else 0.0
                    )
                    open_invested += invested
                    open_current_val += round(curr_p * qty, 2)
                    unrealised_pnl_total += pnl

                portfolio_rows.append(
                    {
                        "Date": pos_date,
                        "Ticker": pos.get("Ticker", sym.replace(".NS", "")),
                        "Company": pos.get("Company", sym),
                        "Status": status,
                        "Remarks / Strategy": pos_remarks,
                        "Entry (₹)": buy_p,
                        "SL (₹)": sl if sl > 0 else "None",
                        "Current Price (₹)": curr_p,
                        "Qty": qty,
                        "Invested (₹)": invested,
                        "P&L (₹)": pnl,
                        "P&L (%)": f"{'+' if pnl >= 0 else ''}{pnl_pct}%",
                    }
                )

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Open Invested Capital", f"₹{open_invested:,.2f}")
            p_col2.metric("Open Portfolio Value", f"₹{open_current_val:,.2f}")
            p_col3.metric(
                "Unrealised P&L (Open)",
                f"₹{unrealised_pnl_total:,.2f}",
                delta=f"{(unrealised_pnl_total / open_invested * 100.0):.2f}%"
                if open_invested > 0
                else "0.00%",
            )
            p_col4.metric(
                "Realised P&L (SL Hit)",
                f"₹{realised_pnl_total:,.2f}",
                delta_color="normal" if realised_pnl_total >= 0 else "inverse",
            )

            st.dataframe(
                pd.DataFrame(portfolio_rows),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🗑️ Reset / Clear All Trades"):
                st.session_state.paper_portfolio = []
                save_portfolio([])
                st.rerun()
        else:
            st.info(
                "No active paper trades. Use the order form above to enter trades with custom remarks & stop loss tracking."
            )
else:
    st.warning("No stocks passed the selected filter criteria.")
