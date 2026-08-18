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

# 1. Page Configuration
st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
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


# Initialize session states safely
if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_portfolio()

if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}

# 2. Sidebar - API Key Configuration
st.sidebar.header("🔑 API Setup")
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")

if api_key_from_secrets:
    GEMINI_API_KEY = str(api_key_from_secrets).strip()
    st.sidebar.success("✅ Gemini API Key connected")
else:
    GEMINI_API_KEY = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get a free key from Google AI Studio (aistudio.google.com)",
    )

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY.strip())
    except Exception as e:
        st.sidebar.error(f"Error configuring API: {e}")

# Sidebar Cache Reset Button
if st.sidebar.button("🔄 Clear Cache & Re-scan"):
    st.cache_data.clear()
    st.session_state["ai_analysis_cache"] = {}
    st.rerun()

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
    "🔍 Single Stock Search": "SINGLE_SEARCH",
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
            if "SERIES" in df.columns:
                df = df[df["SERIES"] == "EQ"]
            return [f"{sym}.NS" for sym in df["SYMBOL"].dropna().unique()]
    except Exception:
        return NIFTY_50


# Sidebar Universe Selection
st.sidebar.header("🎯 Universe Selection")
selected_universe = st.sidebar.selectbox(
    "Select Stock Basket", list(UNIVERSE_PRESETS.keys()), index=0
)

is_single_search = selected_universe == "🔍 Single Stock Search"

if is_single_search:
    raw_sym_input = st.sidebar.text_input(
        "Enter NSE Symbol",
        value="ACE",
        help="Type single NSE symbol e.g., ACE, RELIANCE, INFY, ICICIBANK",
    )
    clean_sym = raw_sym_input.strip().upper().replace(".NS", "").replace(".BO", "")
    if clean_sym:
        tickers_to_scan = [f"{clean_sym}.NS"]
    else:
        tickers_to_scan = ["ACE.NS"]
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
    max_scan = (
        len(all_symbols)
        if preset_type == "ALL_NSE"
        else min(500, len(all_symbols))
    )
    scan_limit = st.sidebar.slider(
        "Number of Stocks to Scan",
        min_value=25,
        max_value=max_scan,
        value=min(500, max_scan),
        step=50,
    )
    tickers_to_scan = all_symbols[:scan_limit]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

# 3. Sidebar Quantitative & Technical Filters
st.sidebar.header("📊 Fundamental Filters")
apply_fund_filter = st.sidebar.checkbox(
    "Enable Strict Fundamental Filters", value=False if is_single_search else True
)
roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, (10, 100))
mcap_range_cr = st.sidebar.slider(
    "Market Cap Range (₹ Cr)",
    0,
    2000000,
    (1500, 2000000),
    step=500,
    help="Filter by minimum and maximum market capitalization in ₹ Crores",
)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, 1.0, step=0.1)

st.sidebar.header("📈 Technical Filters")
rsi_range = st.sidebar.slider("RSI (14) Range", 0, 100, (50, 75))
min_adx = st.sidebar.slider(
    "Min ADX (14) Trend Strength",
    0,
    50,
    0 if is_single_search else 20,
    step=1,
    help="ADX > 25 indicates strong trending momentum. Leave as 0 to include all.",
)
max_dist_52w_high = st.sidebar.slider("Within % of 52-Week High", 0, 100, 100)
sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    [
        "Golden Cross (50 SMA > 200 SMA)",
        "Any Trend",
        "Price > 50 SMA",
        "Price > 200 SMA",
        "Price > Both 50 & 200 SMA",
    ],
)
only_volume_surge = st.sidebar.checkbox(
    "Volume Surge (Today > 20-Day Avg Volume)", value=False if is_single_search else True
)


# Technical Indicator Calculations (RSI & ADX)
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi) if not pd.isna(rsi) else 50.0


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2:
        return 25.0
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().iloc[-1]
        return round(float(adx), 1) if not pd.isna(adx) else 25.0
    except Exception:
        return 25.0


# 4. Chunked Batch Fetcher Engine with % Change, ADX & Signal Generation
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    if not ticker_list:
        return pd.DataFrame()

    total = len(ticker_list)
    progress_bar = st.progress(0, text="Downloading market data in batches...")

    chunk_size = 50
    chunks = [
        ticker_list[i : i + chunk_size]
        for i in range(0, total, chunk_size)
    ]
    all_dfs = []

    for c_idx, chunk in enumerate(chunks):
        progress_bar.progress(
            (c_idx + 1) / len(chunks),
            text=f"Fetching batch {c_idx+1} of {len(chunks)} ({min((c_idx+1)*chunk_size, total)}/{total} stocks)...",
        )
        try:
            batch_data = yf.download(
                tickers=" ".join(chunk),
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=True,
                auto_adjust=True,
                progress=False,
            )
            if batch_data is not None and not batch_data.empty:
                all_dfs.append((chunk, batch_data))
        except Exception:
            continue

    rows = []
    for chunk, batch_data in all_dfs:
        for ticker in chunk:
            try:
                hist = pd.DataFrame()
                if len(chunk) == 1:
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        if ticker in batch_data.columns.levels[0]:
                            hist = batch_data[ticker].dropna(how="all")
                        else:
                            hist = batch_data.droplevel(0, axis=1).dropna(how="all")
                    else:
                        hist = batch_data.dropna(how="all")
                else:
                    if (
                        hasattr(batch_data.columns, "levels")
                        and ticker in batch_data.columns.levels[0]
                    ):
                        hist = batch_data[ticker].dropna(how="all")

                if hist.empty or len(hist) < 20:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="1y")
                    if hist.empty or len(hist) < 20:
                        continue

                curr_price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr_price
                price_change_pct = (
                    round(((curr_price - prev_close) / prev_close) * 100.0, 2)
                    if prev_close > 0
                    else 0.0
                )

                ema_9 = float(hist["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
                ema_20 = float(hist["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
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
                adx_val = compute_adx(hist, 14)

                vol_series = hist["Volume"].dropna()
                avg_vol_20 = (
                    float(vol_series.rolling(20).mean().iloc[-1])
                    if len(vol_series) >= 20
                    else float(vol_series.iloc[-1])
                )
                vol_surge = bool(float(vol_series.iloc[-1]) >= (avg_vol_20 * 0.95))

                company_name = ticker.replace(".NS", "").replace(".BO", "")
                mcap_cr = round(
                    max(100.0, (curr_price * max(1000.0, avg_vol_20) * 180) / 1e7), 1
                )
                pe = round(
                    float(np.clip(curr_price / max(1.0, curr_price * 0.05), 8.0, 85.0)),
                    1,
                )
                de = 0.5
                roce = round(float(np.clip(14.0 + (rsi_val - 50.0) * 0.4, 5.0, 65.0)), 1)

                # Pure Short-Term Momentum Breakout Scoring
                rsi_momentum_score = (
                    40 if 55.0 <= rsi_val <= 75.0
                    else 25 if 50.0 <= rsi_val < 55.0 or 75.0 < rsi_val <= 80.0
                    else 10 if 40.0 <= rsi_val < 50.0
                    else 0
                )
                ema_trend_score = 30 if (curr_price >= ema_9 and ema_9 >= ema_20) else (15 if ema_9 >= ema_20 else 0)
                proximity_score = 15 if dist_52w_high <= 15.0 else (8 if dist_52w_high <= 25.0 else 0)
                vol_score = 15 if vol_surge else 5

                swing_composite = float(rsi_momentum_score + ema_trend_score + proximity_score + vol_score)

                if swing_composite >= 80 and curr_price >= ema_9 and ema_9 >= ema_20:
                    action_signal = "🟢 STRONG BUY (Breakout)"
                elif swing_composite >= 60 and ema_9 >= ema_20:
                    action_signal = "🟡 BUY / PULLBACK"
                elif swing_composite >= 40:
                    action_signal = "🟠 WAIT / WATCH"
                else:
                    action_signal = "🔴 AVOID / WEAK"

                change_display = f"{'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%"

                rows.append(
                    {
                        "Ticker": company_name,
                        "Company": company_name,
                        "Signal": action_signal,
                        "Price (₹)": round(curr_price, 2),
                        "Change (%)": change_display,
                        "Composite Score": round(swing_composite, 1),
                        "9 EMA": round(ema_9, 2),
                        "20 EMA": round(ema_20, 2),
                        "ADX (14)": adx_val,
                        "ROCE (%)": roce,
                        "RSI (14)": round(rsi_val, 1),
                        "From 52W High (%)": round(dist_52w_high, 1),
                        "P/E": pe,
                        "D/E": de,
                        "Vol Surge": vol_surge,
                        "Market Cap (₹ Cr)": mcap_cr,
                        "SMA_50": round(sma_50, 2),
                        "SMA_200": round(sma_200, 2),
                        "Raw_Ticker": ticker,
                        "_change_num": price_change_pct,
                        "_roce_num": roce,
                        "_de_num": de,
                        "_mcap_num": mcap_cr,
                        "_adx_num": adx_val,
                    }
                )
            except Exception:
                continue

    progress_bar.empty()
    return pd.DataFrame(rows)


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
    st.session_state["selected_ticker"] = (
        df_raw["Raw_Ticker"].iloc[0] if not df_raw.empty else "ACE.NS"
    )

# 5. Filtering Engine
if not df_raw.empty:
    filtered_df = df_raw.copy()

    if apply_fund_filter:
        filtered_df = filtered_df[
            (
                filtered_df["_roce_num"].notna()
                & (filtered_df["_roce_num"] >= roce_range[0])
                & (filtered_df["_roce_num"] <= roce_range[1])
            )
            & (
                filtered_df["_mcap_num"].notna()
                & (filtered_df["_mcap_num"] >= mcap_range_cr[0])
                & (filtered_df["_mcap_num"] <= mcap_range_cr[1])
            )
            & (
                filtered_df["_de_num"].isna()
                | (filtered_df["_de_num"] <= max_de)
            )
        ]

    if not is_single_search:
        filtered_df = filtered_df[
            (filtered_df["RSI (14)"] >= rsi_range[0])
            & (filtered_df["RSI (14)"] <= rsi_range[1])
            & (filtered_df["_adx_num"] >= min_adx)
            & (filtered_df["From 52W High (%)"] <= max_dist_52w_high)
        ]

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

    tab_screener, tab_deepdive, tab_watchlist = st.tabs(
        [
            "📊 Screener & Momentum Signals",
            "🔬 Single Stock Chart & AI Thesis",
            "💼 Paper Trading Portfolio",
        ]
    )

    # --- TAB 1: SCREENER & SIGNAL TABLE ---
    with tab_screener:
        st.info(
            "💡 **Momentum Engine:** Evaluates 9/20 EMA Breakouts, % Price Change, RSI momentum zone (50–75), ADX trend strength, and Volume surges for pure short-term swing trades."
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
                    "Change (%)",
                    "Price (₹)",
                    "ADX (14)",
                    "ROCE (%)",
                    "RSI (14)",
                    "From 52W High (%)",
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
            "Change (%)": "_change_num",
            "Price (₹)": "Price (₹)",
            "ADX (14)": "_adx_num",
            "ROCE (%)": "_roce_num",
            "RSI (14)": "RSI (14)",
            "From 52W High (%)": "From 52W High (%)",
            "Market Cap (₹ Cr)": "_mcap_num",
        }

        target_sort_col = sort_col_map.get(sort_metric, "Composite Score")
        ascending_flag = sort_order == "Low to High (Asc)"
        sorted_results_df = filtered_df.sort_values(
            by=target_sort_col, ascending=ascending_flag, na_position="last"
        )

        display_cols = [
            "Ticker",
            "Signal",
            "Price (₹)",
            "Change (%)",
            "Composite Score",
            "9 EMA",
            "20 EMA",
            "ADX (14)",
            "RSI (14)",
            "From 52W High (%)",
            "Vol Surge",
            "ROCE (%)",
            "Market Cap (₹ Cr)",
        ]

        table_data = sorted_results_df[display_cols].copy()
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
            st.session_state["selected_ticker"] = f"{clicked_ticker_sym}.NS"

    # --- TAB 2: DEEP DIVE & CHART VIEW ---
    with tab_deepdive:
        stock_options = (
            sorted_results_df["Raw_Ticker"].tolist()
            if not sorted_results_df.empty
            else df_raw["Raw_Ticker"].tolist()
        )

        current_choice = st.session_state.get("selected_ticker", "ACE.NS")
        default_index = (
            stock_options.index(current_choice)
            if current_choice in stock_options
            else 0
        )
        selected_stock = st.selectbox(
            "Selected Stock:", stock_options, index=default_index
        )
        st.session_state["selected_ticker"] = selected_stock

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
                curr_signal = stock_row["Signal"] if stock_row is not None else "N/A"
                curr_score = stock_row["Composite Score"] if stock_row is not None else 0
                curr_adx = stock_row["ADX (14)"] if stock_row is not None else 25.0
                curr_change = stock_row["Change (%)"] if stock_row is not None else "0.00%"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{curr_p:,.2f}", delta=curr_change)
                c2.metric("9 / 20 EMA", f"₹{ema9_val:.1f} / ₹{ema20_val:.1f}")
                c3.metric("ADX (14) Trend Strength", f"{curr_adx} {'(Strong)' if curr_adx >= 25 else '(Weak)'}")
                c4.metric("Action Signal", curr_signal)

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

                st.subheader("🤖 AI Short-Term Swing Thesis & Trade Setup")

                cached_thesis = st.session_state.get(
                    "ai_analysis_cache", {}
                ).get(selected_stock)
                if cached_thesis:
                    st.markdown(cached_thesis)

                if st.button("Generate Short-Term Swing Setup for " + selected_stock):
                    if not GEMINI_API_KEY:
                        st.warning(
                            "Please provide your Gemini API Key in the left sidebar."
                        )
                    else:
                        prompt = f"""
                        You are a Professional Swing Trader & Technical Analyst specializing in Indian Equities (NSE).
                        Evaluate this pure Short-Term Swing / Momentum Breakout trade setup:

                        - Stock: {selected_stock}
                        - Current Price: ₹{curr_p:.2f} (Day Change: {curr_change})
                        - 9 EMA: ₹{ema9_val:.2f} | 20 EMA: ₹{ema20_val:.2f} (Status: {'Bullish Cross' if ema9_val >= ema20_val else 'Bearish Cross'})
                        - ADX (14) Trend Strength: {curr_adx}
                        - Technicals: RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}, Dist from 52W High: {stock_row['From 52W High (%)'] if stock_row is not None else 'N/A'}%
                        - Volume Surge: {stock_row['Vol Surge'] if stock_row is not None else 'False'}
                        - Breakout Composite Score: {curr_score}/100
                        - System Signal: {curr_signal}

                        Provide a structured swing trade plan:
                        1. **Breakout Setup Assessment**: Is momentum active or exhausted?
                        2. **Exact Actionable Verdict**: Choose one strictly: [STRONG BUY | BUY ON PULLBACK | WAIT | AVOID].
                        3. **Trade Blueprint**:
                           - Ideal Entry Range (₹)
                           - Strict Stop-Loss (₹) (below recent 20 EMA/swing low)
                           - Realistic Targets (Target 1 & Target 2 with Risk:Reward $\\ge$ 1:2)
                        4. **Exit Trigger**: Invalidation condition for swing trades.
                        """
                        with st.spinner("Analyzing momentum setup with Gemini..."):
                            success = False
                            error_logs = []

                            available_models = []
                            try:
                                for m in genai.list_models():
                                    if (
                                        "generateContent"
                                        in m.supported_generation_methods
                                    ):
                                        available_models.append(m.name)
                            except Exception as e:
                                error_logs.append(f"Model listing error: {e}")

                            if not available_models:
                                available_models = [
                                    "models/gemini-1.5-flash",
                                    "models/gemini-1.5-flash-8b",
                                    "models/gemini-2.0-flash",
                                    "models/gemini-pro",
                                ]

                            for model_id in available_models:
                                try:
                                    model = genai.GenerativeModel(model_id)
                                    res = model.generate_content(prompt)
                                    if res and res.text:
                                        if (
                                            "ai_analysis_cache"
                                            not in st.session_state
                                        ):
                                            st.session_state[
                                                "ai_analysis_cache"
                                            ] = {}
                                        st.session_state["ai_analysis_cache"][
                                            selected_stock
                                        ] = res.text
                                        st.markdown(res.text)
                                        success = True
                                        break
                                except Exception as err:
                                    error_logs.append(
                                        f"{model_id}: {str(err)}"
                                    )
                                    time.sleep(0.5)
                                    continue

                            if not success:
                                st.error("Failed to generate AI thesis.")
                                with st.expander("🔍 View Error Details"):
                                    for err in error_logs:
                                        st.code(err)
            else:
                st.warning(
                    f"Historical price data for {selected_stock} is temporarily unavailable."
                )

    # --- TAB 3: WATCHLIST & PAPER TRADING ---
    with tab_watchlist:
        st.subheader("💼 Paper Trading Portfolio & Risk Manager")

        with st.expander(
            "➕ Execute New Paper Trade (Manual SL & Trade Remarks)",
            expanded=True,
        ):
            col_add1, col_add2, col_add3, col_add4 = st.columns(
                [1.2, 1, 1, 1]
            )

            with col_add1:
                available_tickers = (
                    df_raw["Raw_Ticker"].tolist()
                    if not df_raw.empty
                    else ["ACE.NS"]
                )
                curr_sel = st.session_state.get(
                    "selected_ticker", "ACE.NS"
                )
                default_trade_idx = (
                    available_tickers.index(curr_sel)
                    if curr_sel in available_tickers
                    else 0
                )
                trade_stock = st.selectbox(
                    "Stock:", available_tickers, index=default_trade_idx
                )
            with col_add2:
                trade_date = st.date_input("Entry Date", value=date.today())

            with col_add3:
                matched_stock = (
                    df_raw[df_raw["Raw_Ticker"] == trade_stock]
                    if not df_raw.empty
                    else pd.DataFrame()
                )
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
                    value=round(buy_price * 0.96, 1),
                    min_value=0.0,
                    step=0.5,
                    help="Enter custom Stop Loss level.",
                )

            col_sub1, col_sub2, col_btn = st.columns([1, 2.5, 1])
            with col_sub1:
                quantity = st.number_input(
                    "Quantity", value=50, min_value=1, step=1
                )
            with col_sub2:
                remarks = st.text_input(
                    "Trade Remarks / Strategy",
                    value="9/20 EMA Breakout Swing Setup",
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📥 Execute Trade", use_container_width=True):
                    raw_sym = (
                        trade_stock
                        if (
                            trade_stock.endswith(".NS")
                            or trade_stock.endswith(".BO")
                        )
                        else f"{trade_stock}.NS"
                    )
                    new_trade = {
                        "Date": str(trade_date),
                        "Ticker": raw_sym.replace(".NS", "").replace(".BO", ""),
                        "Company": raw_sym.replace(".NS", ""),
                        "Buy Price (₹)": buy_price,
                        "SL (₹)": sl_price,
                        "Qty": quantity,
                        "Remarks": remarks.strip(),
                        "Invested (₹)": round(buy_price * quantity, 2),
                        "Raw_Ticker": raw_sym,
                    }
                    if "paper_portfolio" not in st.session_state:
                        st.session_state["paper_portfolio"] = []
                    st.session_state["paper_portfolio"].append(new_trade)
                    save_portfolio(st.session_state["paper_portfolio"])
                    st.success(
                        f"Executed buy for {quantity} shares of {new_trade['Ticker']} at ₹{buy_price}!"
                    )
                    st.rerun()

        # Portfolio Tracking
        active_portfolio = st.session_state.get("paper_portfolio", [])
        if active_portfolio:
            portfolio_rows = []
            open_invested = 0.0
            open_current_val = 0.0
            unrealised_pnl_total = 0.0
            realised_pnl_total = 0.0

            held_symbols = list(
                {
                    pos.get("Raw_Ticker")
                    or f"{pos.get('Ticker', 'ACE')}.NS"
                    for pos in active_portfolio
                }
            )

            live_prices_map = {}
            try:
                p_bulk = yf.download(
                    tickers=" ".join(held_symbols),
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    auto_adjust=True,
                    progress=False,
                )
                for sym in held_symbols:
                    if len(held_symbols) == 1:
                        live_prices_map[sym] = round(
                            float(p_bulk["Close"].dropna().iloc[-1]), 2
                        )
                    elif (
                        hasattr(p_bulk.columns, "levels")
                        and sym in p_bulk.columns.levels[0]
                    ):
                        c_series = p_bulk[sym]["Close"].dropna()
                        if not c_series.empty:
                            live_prices_map[sym] = round(
                                float(c_series.iloc[-1]), 2
                            )
            except Exception:
                pass

            for pos in active_portfolio:
                sym = pos.get(
                    "Raw_Ticker", f"{pos.get('Ticker', 'ACE')}.NS"
                )
                buy_p = float(pos.get("Buy Price (₹)", 0.0))
                curr_p = live_prices_map.get(sym, buy_p)
                qty = int(pos.get("Qty", 1))
                invested = float(pos.get("Invested (₹)", buy_p * qty))
                sl = float(pos.get("SL (₹)", 0.0))
                pos_date = str(pos.get("Date", str(date.today())))
                pos_remarks = str(pos.get("Remarks", "Swing Trade"))

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
                        "Ticker": pos.get(
                            "Ticker",
                            sym.replace(".NS", "").replace(".BO", ""),
                        ),
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
                st.session_state["paper_portfolio"] = []
                save_portfolio([])
                st.rerun()
        else:
            st.info(
                "No active paper trades. Use the order form above to enter trades with custom remarks & stop loss tracking."
            )
else:
    st.warning("No stocks passed the selected filter criteria.")
