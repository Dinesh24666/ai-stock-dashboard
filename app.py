import os
import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. Page Config
st.set_page_config(
    page_title="Indian Market Screener",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Indian Market AI Screener & Dashboard")

# 2. Sidebar - Setup & Preset Universes
st.sidebar.header("⚙️ Configuration")
GEMINI_API_KEY = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Get a free key from Google AI Studio (aistudio.google.com)",
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())

# Comprehensive NSE Sector Baskets
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
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
        "NMDC.NS", "SAIL.NS", "NATIONALUM.NS"
    ],
    "Defence, Rail & PSUs": [
        "HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS",
        "IRFC.NS", "COCHINSHIP.NS", "BDL.NS", "CONCOR.NS"
    ],
    "Infrastructure & Realty": [
        "LT.NS", "DLF.NS", "GODREJPROP.NS", "MACROTECH.NS", "OBEROIRLTY.NS",
        "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "NCC.NS"
    ],
    "Custom Watchlist": [],
}

selected_universe = st.sidebar.selectbox("Select Stock Basket", list(UNIVERSE_PRESETS.keys()))

if selected_universe == "Custom Watchlist":
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

# 3. Screener Sliders & Filters
st.sidebar.header("🎯 Quantitative Filters")
min_roe = st.sidebar.slider("Min Return on Equity (ROE %)", 0, 40, 10)
max_pe = st.sidebar.slider("Max P/E Ratio", 5, 120, 60)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 3.5, 2.0, step=0.1)
only_bullish_sma = st.sidebar.checkbox("Only Bullish Trend (Price >= 50 & 200 SMA)", value=False)


# 4. Cached Batch Data Fetcher
@st.cache_data(ttl=1800)
def fetch_screener_universe(ticker_list):
    rows = []
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist is None or hist.empty or len(hist) < 20:
                continue

            info = t.info
            curr_price = hist["Close"].iloc[-1]
            sma_50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else curr_price
            sma_200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else curr_price

            roe = (info.get("returnOnEquity") or 0.10) * 100.0
            pe = info.get("trailingPE") or info.get("forwardPE") or 25.0
            de = (info.get("debtToEquity") or 50.0) / 100.0
            mcap_cr = (info.get("marketCap") or 0) / 1e7
            fcf = info.get("freeCashflow") or 0
            rev = info.get("totalRevenue") or 1
            fcf_margin = (fcf / rev) * 100.0 if rev > 0 else 0.0

            # Composite Score (0-100)
            fund_score = min(100, max(0, (roe / 25.0) * 60 + ((2.0 - min(de, 2.0)) / 2.0) * 40))
            tech_score = (
                (35 if curr_price >= sma_50 else 0)
                + (35 if curr_price >= sma_200 else 0)
                + (30 if sma_50 >= sma_200 else 0)
            )
            composite = round(0.6 * fund_score + 0.4 * tech_score, 1)

            rows.append({
                "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "Company": info.get("shortName") or info.get("longName") or ticker.replace(".NS", ""),
                "Price (₹)": round(curr_price, 2),
                "Composite Score": composite,
                "P/E": round(pe, 1),
                "ROE (%)": round(roe, 1),
                "D/E": round(de, 2),
                "FCF Margin (%)": round(fcf_margin, 1),
                "Market Cap (₹ Cr)": round(mcap_cr, 0),
                "SMA_50": round(sma_50, 2),
                "SMA_200": round(sma_200, 2),
                "Above 200 SMA": bool(curr_price >= sma_200),
                "Raw_Ticker": ticker,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# Fetch Data
with st.spinner(f"Screening {len(tickers_to_scan)} stocks in basket..."):
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
        st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)} passed filters)")

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
                        - Price: ₹{hist['Close'].iloc[-1]:.2f}
                        - P/E: {stock_row['P/E'] if stock_row is not None else 'N/A'} | ROE: {stock_row['ROE (%)'] if stock_row is not None else 'N/A'}% | Debt/Equity: {stock_row['D/E'] if stock_row is not None else 'N/A'}
                        - Quality Composite Score: {stock_row['Composite Score'] if stock_row is not None else 'N/A'}/100

                        Provide a Screener.in-style analyst breakdown:
                        1. **Pros & Competitive Moat**
                        2. **Key Risks & Red Flags**
                        3. **Investment Verdict** (Bullish / Neutral / Bearish)
                        """
                        with st.spinner("Generating AI Analysis..."):
                            try:
                                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                                res = model.generate_content(prompt)
                                st.markdown(res.text)
                            except Exception as e:
                                try:
                                    fallback = genai.GenerativeModel("gemini-2.0-flash")
                                    st.markdown(fallback.generate_content(prompt).text)
                                except Exception as err:
                                    st.error(f"Error generating AI thesis: {err}")
else:
    st.warning("No data retrieved. Verify your stock basket selection or network connection.")
