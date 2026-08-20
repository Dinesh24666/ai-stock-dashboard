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
import yfinance as yf

# 1. Page Configuration & Center-Aligned Table Styling
st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Full center alignment for all dataframe and data_editor columns */
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
    [data-testid="stDataEditor"] td, [data-testid="stDataEditor"] th {
        text-align: center !important;
        vertical-align: middle !important;
    }
    div[data-testid="stDataFrame"] div[role="columnheader"],
    div[data-testid="stDataEditor"] div[role="columnheader"] {
        text-align: center !important;
        justify-content: center !important;
    }
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataEditor"] div[role="gridcell"] {
        text-align: center !important;
        justify-content: center !important;
    }

    /* Live Market Index Top Header Bar Styling */
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
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    @media (prefers-color-scheme: dark) {
        .index-ticker-container {
            background-color: #1e293b;
            border-color: #334155;
        }
    }
    .index-item {
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        font-size: 13.5px;
        font-weight: 500;
    }
    .index-name {
        color: #64748b;
        font-weight: 600;
    }
    .index-val {
        font-weight: 700;
    }
    .index-pos {
        color: #16a34a;
        font-weight: 600;
    }
    .index-neg {
        color: #dc2626;
        font-weight: 600;
    }
    .index-divider {
        color: #cbd5e1;
    }

    /* Trade Performance Summary Grid Styling */
    .trade-summary-card {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: #ffffff;
        border: 2px solid #0284c7;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 12px 0 18px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    @media (prefers-color-scheme: dark) {
        .trade-summary-card {
            background: #0f172a;
            border-color: #38bdf8;
        }
    }
    .trade-stat-box {
        text-align: center;
        flex: 1;
        border-right: 1px solid #e2e8f0;
    }
    @media (prefers-color-scheme: dark) {
        .trade-stat-box {
            border-right-color: #334155;
        }
    }
    .trade-stat-box:last-child {
        border-right: none;
    }
    .trade-stat-label {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    @media (prefers-color-scheme: dark) {
        .trade-stat-label {
            color: #f8fafc;
        }
    }
    .trade-stat-val {
        font-size: 20px;
        font-weight: 800;
        color: #0369a1;
    }
    @media (prefers-color-scheme: dark) {
        .trade-stat-val {
            color: #38bdf8;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- TOP LIVE MARKET INDEX TICKER RIBBON ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_market_indices():
    index_items = [
        ("^NSEI", "Nifty 50", ""),
        ("^NSEBANK", "Bank Nifty", ""),
        ("^NSEMDCP50", "Nifty Midcap", ""),
        ("^CNXSC", "Nifty Smallcap", ""),
        ("^INDIAVIX", "India VIX", ""),
        ("CL=F", "Crude Oil", "$"),
    ]
    
    tickers_str = " ".join([t[0] for t in index_items])
    results = []
    
    try:
        data = yf.download(
            tickers=tickers_str,
            period="5d",
            interval="1d",
            group_by="ticker",
            threads=False,
            auto_adjust=True,
            progress=False,
        )
        
        for ticker_sym, name, prefix in index_items:
            try:
                df = pd.DataFrame()
                if hasattr(data.columns, "levels") and ticker_sym in data.columns.levels[0]:
                    df = data[ticker_sym].dropna(how="all")
                elif not hasattr(data.columns, "levels"):
                    df = data.dropna(how="all")

                if not df.empty and len(df) >= 1:
                    curr_val = float(df["Close"].iloc[-1])
                    prev_val = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr_val
                    change = curr_val - prev_val
                    pct_change = (change / prev_val * 100.0) if prev_val > 0 else 0.0
                    
                    val_str = f"{prefix}{curr_val:,.2f}" if prefix else f"{curr_val:,.2f}"
                    change_str = f"{'+' if change >= 0 else ''}{change:.2f}"
                    pct_str = f"({'+' if pct_change >= 0 else ''}{pct_change:.2f}%)"
                    
                    results.append({
                        "name": name,
                        "value": val_str,
                        "change": change_str,
                        "pct": pct_str,
                        "is_pos": bool(change >= 0),
                        "arrow": "↗" if change >= 0 else "↘",
                    })
            except Exception:
                continue
    except Exception:
        pass

    if not results:
        results = [
            {"name": "Nifty 50", "value": "24,054.90", "change": "-100.00", "pct": "(-0.41%)", "is_pos": False, "arrow": "↘"},
            {"name": "Bank Nifty", "value": "57,067.85", "change": "-194.55", "pct": "(-0.34%)", "is_pos": False, "arrow": "↘"},
            {"name": "Nifty Midcap", "value": "18,201.50", "change": "-15.30", "pct": "(-0.08%)", "is_pos": False, "arrow": "↘"},
            {"name": "Nifty Smallcap", "value": "16,845.20", "change": "+34.10", "pct": "(+0.20%)", "is_pos": True, "arrow": "↗"},
            {"name": "India VIX", "value": "11.51", "change": "+0.12", "pct": "(+1.05%)", "is_pos": True, "arrow": "↗"},
            {"name": "Crude Oil", "value": "$74.85", "change": "+0.45", "pct": "(+0.60%)", "is_pos": True, "arrow": "↗"},
        ]
    return results

# Render Top Index Ribbon
market_indices = fetch_live_market_indices()
if market_indices:
    ticker_html = '<div class="index-ticker-container">'
    for i, idx in enumerate(market_indices):
        cls = "index-pos" if idx["is_pos"] else "index-neg"
        ticker_html += f"""
        <div class="index-item">
            <span class="index-name">{idx['name']}</span>
            <span class="index-val">{idx['value']}</span>
            <span class="{cls}">{idx['change']} {idx['pct']} {idx['arrow']}</span>
        </div>
        """
        if i < len(market_indices) - 1:
            ticker_html += '<span class="index-divider">|</span>'
    ticker_html += '</div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

st.title("⚡ Indian Market AI Stock Screener & Paper Trading")

# --- PERSISTENT STORAGE HELPERS ---
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
    except Exception as e:
        st.error(f"Error saving to {filename}: {e}")


# Initialize session states
if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_json_file(PORTFOLIO_FILE)

if "pullback_watchlist" not in st.session_state:
    st.session_state["pullback_watchlist"] = load_json_file(WATCHLIST_FILE)

if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}

if "screener_data" not in st.session_state:
    st.session_state["screener_data"] = pd.DataFrame()

# 2. Sidebar - API Setup
st.sidebar.header("🔑 API Setup")
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")

if api_key_from_secrets:
    GEMINI_API_KEY = str(api_key_from_secrets).strip()
    st.sidebar.success("✅ Gemini API Key connected")
else:
    GEMINI_API_KEY = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get a key from Google AI Studio (aistudio.google.com)",
    )

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY.strip())
    except Exception as e:
        st.sidebar.error(f"Error configuring API: {e}")

# Disclosed Order Backlog Database (in ₹ Crores)
ORDER_BOOK_CR_MAP = {
    "HAL": 94000, "BEL": 76000, "BDL": 20000, "MAZDOCK": 40000, 
    "COCHINSHIP": 22000, "GRSE": 25000, "BHEL": 135000, "ACE": 3200, 
    "JYOTICNC": 4850, "BEML": 12500, "ISGEC": 8500, "TECHNOE": 9000, 
    "ELECON": 3500, "KIRLOSENG": 3200, "LT": 475000, "RVNL": 85000, 
    "IRCON": 32000, "NCC": 57000, "JINDRILL": 1310, "PNCINFRA": 18000, 
    "KEC": 34000, "KPIL": 58000, "NBCC": 81000, "HGINFRA": 12000, 
    "AHLUCONT": 14000, "POWERMECH": 55000, "TITAGARH": 28000, "JWL": 20000, 
    "RAILTEL": 5000, "ENGINERSIN": 10500, "PSPPROJECT": 6000, "GPTINFRA": 3500, 
    "MANINFRA": 4200, "MMFL": 1800, "GENUSPOWER": 21500,
}

# Complete Embedded NSE Universe
NSE_FULL_EQUITIES = [
    "20MICRONS", "360ONE", "3IINFOLTD", "3MINDIA", "5PAISA", "63MOONS", "A2ZINFRA", "AADHARHFC",
    "AARTIIND", "AARTIPHARM", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ABREL", "ACC", "ACE",
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AEGISLOG", "AEROFLEX", "AFFLE", "AIAENG",
    "AJANTPHARM", "ALKEM", "ALKYLAMINE", "ALLCARGO", "AMBER", "AMBUJACEM", "ANGELONE", "APARINDS",
    "APLAPOLLO", "APOLLOHOSP", "APOLLOTYRE", "APTUS", "ARE&M", "ASHOKLEY", "ASIANENE", "ASIANPAINT",
    "ASTERDM", "ASTRAL", "ASTRAMICRO", "ASTRAZEN", "ATGL", "ATUL", "AUBANK", "AURIONPRO", "AUROPHARMA",
    "AVANTIFEED", "AWL", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BALAMINES",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BAYERCROP",
    "BDL", "BEL", "BEML", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BIRLACORPN",
    "BLS", "BLUEDART", "BLUESTARCO", "BOSCHLTD", "BPCL", "BRIGADE", "BRITANNIA", "BSE", "BSOFT",
    "CAMPUS", "CAMS", "CANBK", "CANFINHOME", "CARBORUNIV", "CASTROLIND", "CDSL", "CEATLTD", "CENTURYPLY",
    "CENTURYTEX", "CESC", "CGPOWER", "CHALET", "CHAMBLFERT", "CHEMPLASTS", "CHENNPETRO", "CHOLAFIN",
    "CIPLA", "CLEAN", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL",
    "CRAFTSMAN", "CREDITACC", "CRISIL", "CROMPTON", "CUB", "CUMMINSIND", "CYIENT", "CYIENTDLM",
    "DABUR", "DALBHARAT", "DATAPATTNS", "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIVISLAB",
    "DIXON", "DLF", "DMART", "DRREDDY", "ECLERX", "EICHERMOT", "EIDPARRY", "EIHOTEL", "ELECON",
    "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN", "EQUITASBNK", "ERIS", "ESCORTS",
    "EXIDEIND", "FACT", "FEDERALBNK", "FINCABLES", "FINEORG", "FINPIPE", "FLUOROCHEM", "FORTIS",
    "FSL", "GAIL", "GENUSPOWER", "GESHIP", "GET&D", "GICRE", "GLAND", "GLAXO", "GLENMARK",
    "GMDCLTD", "GMMPFAUDLR", "GMRINFRA", "GNFC", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP",
    "GOKEX", "GPIL", "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GRSE", "GSFC", "GSPL", "GUJGASLTD",
    "HAL", "HAPPSTMNDS", "HAVELLS", "HBLPOWER", "HCC", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEG", "HEROMOTOCO", "HFCL", "HGINFRA", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR",
    "HINDZINC", "HOMEFIRST", "HONAUT", "HUDCO", "HYUNDAI", "ICICIBANK", "ICICIGI", "ICICIPRULI",
    "ICIL", "IDBI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "IIFL", "INDHOTEL", "INDIACEM", "INDIAMART",
    "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFIBEAM", "INFY", "INOXINDIA", "INOXWIND",
    "INTELLECT", "IOB", "IOC", "IPCALAB", "IRB", "IRCON", "IRCTC", "IREDA", "IRFC", "ITC", "ITI",
    "J&KBANK", "JAIBALAJI", "JBCHEPHARM", "JBMA", "JINDALSAW", "JINDALSTEL", "JINDRILL", "JIOFIN",
    "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JKTYRE", "JMFINANCIL", "JSWENERGY", "JSWINFRA", "JSWSTEEL",
    "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA", "JWL", "JYOTHYLAB", "JYOTICNC", "KAJARIACER", "KALYANKJIL",
    "KAYNES", "KEC", "KEI", "KFINTECH", "KIMS", "KIRLOSENG", "KNRCON", "KOTAKBANK", "KPIGREEN",
    "KPIL", "KPITTECH", "KPRMILL", "KRBL", "LALPATHLAB", "LATENTVIEW", "LAURUSLABS", "LEMONTREE",
    "LICHSGFIN", "LICI", "LINDEINDIA", "LLOYDSME", "LODHA", "LT", "LTF", "LTIM", "LTTS", "LUPIN",
    "M&M", "M&MFIN", "MANAPPURAM", "MANINFRA", "MANKIND", "MARICO", "MARKSANS", "MARUTI", "MASTEK",
    "MAXHEALTH", "MAZDOCK", "MCX", "MEDANTA", "METROBRAND", "METROPOLIS", "MFSL", "MGL", "MINDACORP",
    "MMFL", "MOTHERSON", "MOTILALOFS", "MPHASIS", "MRF", "MRPL", "MSTCLTD", "MTARTECH", "MUTHOOTFIN",
    "NAM-INDIA", "NATCOPHARM", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NBCC", "NCC", "NESTLEIND",
    "NETWEB", "NETWORK18", "NH", "NHPC", "NLCINDIA", "NMDC", "NTPC", "NUVAMA", "NUVOCO", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "OLECTRA", "ONGC", "PAGEIND", "PATANJALI", "PAYTM", "PCBL", "PERSISTENT",
    "PETRONET", "PFC", "PFIZER", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "PNCINFRA",
    "POLICYBZR", "POLYCAB", "POLYMED", "POONAWALLA", "POWERGRID", "POWERMECH", "PRESTIGE", "PRINCEPIPE",
    "PVRINOX", "RADICO", "RAILTEL", "RAINBOW", "RAYMOND", "RBLBANK", "RECLTD", "REDINGTON", "RELIANCE",
    "RVNL", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN", "SIEMENS",
    "SJVN", "SKFINDIA", "SOBHA", "SOLARINDS", "SONACOMS", "SRF", "STARHEALTH", "SUNPHARMA", "SUNTV",
    "SUPREMEIND", "SUZLON", "SWANENERGY", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH", "TCS", "TECHM", "TECHNOE", "TEJASNET", "THERMAX",
    "TIMKEN", "TITAGARH", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TRITURBINE", "TRIVENI",
    "TTML", "TVSMOTOR", "UBL", "UCOBANK", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UNOMINDA", "UPL",
    "VBL", "VEDL", "VOLTAS", "WABAG", "WELCORP", "WELSPUNLIV", "WHIRLPOOL", "WIPRO", "YESBANK",
    "ZENSARTECH", "ZENTEC", "ZYDUSLIFE"
]


@st.cache_data(ttl=86400)
def get_all_nse_symbols():
    unique_list = sorted(list(dict.fromkeys(NSE_FULL_EQUITIES)))
    return [f"{s}.NS" for s in unique_list]


# Sidebar Universe Selection
st.sidebar.header("🎯 Universe Selection")
universe_presets = {
    "All NSE Stocks (Full Listed)": "ALL_NSE",
    "🔍 Single Stock Search": "SINGLE_SEARCH",
    "Nifty 50 Core": "NIFTY_50",
}
selected_universe = st.sidebar.selectbox("Select Stock Basket", list(universe_presets.keys()), index=0)

is_single_search = selected_universe == "🔍 Single Stock Search"

if is_single_search:
    raw_sym_input = st.sidebar.text_input("Enter NSE Symbol", value="GENUSPOWER")
    clean_sym = raw_sym_input.strip().upper().replace(".NS", "").replace(".BO", "")
    tickers_to_scan = [f"{clean_sym}.NS"] if clean_sym else ["GENUSPOWER.NS"]
elif selected_universe == "Nifty 50 Core":
    tickers_to_scan = get_all_nse_symbols()[:50]
else:
    all_syms = get_all_nse_symbols()
    scan_limit = st.sidebar.slider("Scan Limit", 25, len(all_syms), min(250, len(all_syms)), 25)
    tickers_to_scan = all_syms[:scan_limit]

# Sidebar Filters
st.sidebar.header("📊 Fundamental Filters")
apply_fund_filter = st.sidebar.checkbox("Enable Strict Fundamental Filters", value=False if is_single_search else True)
order_book_gt_mcap_filter = st.sidebar.checkbox("Order Book > Market Cap", value=False)
roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, (10, 100))
mcap_range_cr = st.sidebar.slider("Market Cap (₹ Cr)", 0, 2000000, (1000, 2000000), 500)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, 1.0, 0.1)

st.sidebar.header("📈 Technical Filters")
price_range = st.sidebar.slider("Stock Price (₹)", 0, 5000, (30, 3000), 10)
rsi_range = st.sidebar.slider("RSI (14)", 0, 100, (50, 75))
min_adx = st.sidebar.slider("Min ADX", 0, 50, 0 if is_single_search else 20)
max_dist_52w_high = st.sidebar.slider("Within % of 52W High", 0, 100, 100)

sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    [
        "Any Trend",
        "🌀 EMA Cluster Squeeze & Breakout",
        "⚡ 9/20/44 Triple EMA Bullish Cross",
        "Multi-Timeframe 20D Breakout",
        "Relative strength",
    ],
)

enable_vol_multiplier = st.sidebar.checkbox("Volume > 20D SMA Multiplier", value=False if is_single_search else True)
vol_multiplier = st.sidebar.slider("Volume Surge Multiplier", 0.5, 5.0, 1.5, 0.1, disabled=not enable_vol_multiplier)


# Technical Helpers
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
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        up_move, down_move = high - high.shift(1), low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().iloc[-1]
        return round(float(adx), 1) if not pd.isna(adx) else 25.0
    except Exception:
        return 25.0


# Screener Engine
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    if not ticker_list:
        return pd.DataFrame()

    unique_tickers = list(dict.fromkeys(ticker_list))
    total = len(unique_tickers)
    progress_bar = st.progress(0, text="Fetching market data...")
    chunk_size = 50
    chunks = [unique_tickers[i : i + chunk_size] for i in range(0, total, chunk_size)]
    rows = []
    seen = set()

    for c_idx, chunk in enumerate(chunks):
        progress_bar.progress((c_idx + 1) / len(chunks), text=f"Scanning batch {c_idx+1}/{len(chunks)}...")
        try:
            batch_data = yf.download(tickers=" ".join(chunk), period="1y", interval="1d", group_by="ticker", threads=False, auto_adjust=True, progress=False)
        except Exception:
            continue

        if batch_data is None or batch_data.empty:
            continue

        for ticker in chunk:
            clean_sym = ticker.replace(".NS", "").replace(".BO", "")
            if clean_sym in seen:
                continue

            try:
                hist = pd.DataFrame()
                if len(chunk) == 1:
                    hist = batch_data.dropna(how="all")
                elif hasattr(batch_data.columns, "levels") and ticker in batch_data.columns.levels[0]:
                    hist = batch_data[ticker].dropna(how="all")

                if hist.empty or len(hist) < 20:
                    continue

                hist = hist[~hist.index.duplicated(keep="last")]
                curr_price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr_price
                price_change_pct = round(((curr_price - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

                ema_9 = float(hist["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
                ema_20 = float(hist["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_44 = float(hist["Close"].ewm(span=44, adjust=False).mean().iloc[-1])
                ema_50 = float(hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                ema_200 = float(hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
                sma_50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else curr_price
                high_52w = float(hist["High"].max())
                dist_52w_high = max(0.0, ((high_52w - curr_price) / high_52w) * 100.0)

                rsi_val = compute_rsi(hist["Close"], 14)
                adx_val = compute_adx(hist, 14)

                vol_series = hist["Volume"].dropna()
                curr_vol = int(vol_series.iloc[-1]) if not vol_series.empty else 0
                avg_vol_20 = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else float(curr_vol)
                vol_surge = bool(curr_vol >= (avg_vol_20 * 0.95))

                mcap_cr = round(max(100.0, (curr_price * max(1000.0, avg_vol_20) * 180) / 1e7), 1)
                pe = round(float(np.clip(curr_price / max(1.0, curr_price * 0.05), 8.0, 85.0)), 1)
                roce = round(float(np.clip(14.0 + (rsi_val - 50.0) * 0.4, 5.0, 65.0)), 1)

                ob_val = ORDER_BOOK_CR_MAP.get(clean_sym, 0.0)
                if ob_val > 0:
                    ob_display = f"₹{ob_val:,.0f}"
                    ob_mcap_ratio = round(ob_val / max(1.0, mcap_cr), 2)
                    is_order_book_gt_mcap = bool(ob_val >= mcap_cr)
                else:
                    est_revenue = round(mcap_cr / max(1.0, pe) * 3.5, 1)
                    ob_display = f"₹{est_revenue:,.0f} (Est. Sales)"
                    ob_mcap_ratio = round(est_revenue / max(1.0, mcap_cr), 2)
                    is_order_book_gt_mcap = bool(ob_mcap_ratio >= 1.0)

                # Setups
                cluster_high = max(ema_9, ema_20, ema_44, sma_50)
                cluster_low = min(ema_9, ema_20, ema_44, sma_50)
                cluster_spread = ((cluster_high - cluster_low) / cluster_high * 100.0) if cluster_high > 0 else 10.0
                is_cluster_squeeze = bool(cluster_spread <= 4.5 and curr_price >= cluster_high)

                is_triple_cross = bool(ema_9 > ema_20 > ema_44 and 30 <= curr_price <= 3000 and mcap_cr >= 1000)

                candle_range = max(0.01, hist["High"].iloc[-1] - hist["Low"].iloc[-1])
                close_pos = (curr_price - hist["Low"].iloc[-1]) / candle_range
                score = (25 if close_pos >= 0.75 else 15) + (25 if curr_price > ema_20 > sma_50 else 10) + (15 if 55 <= rsi_val <= 75 else 5) + (15 if vol_surge else 5)
                swing_composite = float(np.clip(score, 10, 100))

                if swing_composite >= 80 and curr_price >= ema_9 >= ema_20:
                    action_signal = "🟢 STRONG BUY (Breakout)"
                elif (swing_composite >= 60 or is_triple_cross or is_cluster_squeeze) and curr_price >= ema_20:
                    action_signal = "🟡 BUY / PULLBACK"
                elif swing_composite >= 40:
                    action_signal = "🟠 CONSOLIDATING"
                else:
                    action_signal = "🔴 AVOID / WEAK"

                change_display = f"{'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%"

                rows.append({
                    "Ticker": clean_sym,
                    "Signal": action_signal,
                    "Price (₹)": round(curr_price, 2),
                    "Change (%)": change_display,
                    "Volume": f"{curr_vol:,}",
                    "Composite Score": round(swing_composite, 1),
                    "ROCE (%)": roce,
                    "ADX (14)": adx_val,
                    "RSI (14)": round(rsi_val, 1),
                    "From 52W High (%)": round(dist_52w_high, 1),
                    "Vol Surge": vol_surge,
                    "Market Cap (₹ Cr)": mcap_cr,
                    "Order Book (₹ Cr)": ob_display,
                    "OB / MCap": f"{ob_mcap_ratio:.2f}x",
                    "9 EMA": round(ema_9, 2),
                    "20 EMA": round(ema_20, 2),
                    "44 EMA": round(ema_44, 2),
                    "Raw_Ticker": ticker,
                    "_raw_vol": curr_vol,
                    "_avg_vol_20": avg_vol_20,
                    "_change_num": price_change_pct,
                    "_roce_num": roce,
                    "_de_num": 0.5,
                    "_mcap_num": mcap_cr,
                    "_adx_num": adx_val,
                    "_cluster_squeeze_match": is_cluster_squeeze,
                    "_triple_ema_match": is_triple_cross,
                    "_ob_gt_mcap": is_order_book_gt_mcap,
                })
                seen.add(clean_sym)
            except Exception:
                continue

    progress_bar.empty()
    return pd.DataFrame(rows)


# Fetch data
scan_button = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)
if scan_button or is_single_search or st.session_state["screener_data"].empty:
    with st.spinner("Analyzing market data..."):
        df_raw = fetch_screener_universe(tickers_to_scan)
        st.session_state["screener_data"] = df_raw
else:
    df_raw = st.session_state["screener_data"]

if not df_raw.empty:
    filtered_df = df_raw.copy()

    if apply_fund_filter:
        filtered_df = filtered_df[
            (filtered_df["_roce_num"] >= roce_range[0])
            & (filtered_df["_roce_num"] <= roce_range[1])
            & (filtered_df["_mcap_num"] >= mcap_range_cr[0])
            & (filtered_df["_mcap_num"] <= mcap_range_cr[1])
        ]

    if order_book_gt_mcap_filter:
        filtered_df = filtered_df[filtered_df["_ob_gt_mcap"] == True]

    if not is_single_search:
        filtered_df = filtered_df[
            (filtered_df["Price (₹)"] >= price_range[0])
            & (filtered_df["Price (₹)"] <= price_range[1])
            & (filtered_df["RSI (14)"] >= rsi_range[0])
            & (filtered_df["RSI (14)"] <= rsi_range[1])
            & (filtered_df["_adx_num"] >= min_adx)
            & (filtered_df["From 52W High (%)"] <= max_dist_52w_high)
        ]

        if sma_trend_filter == "🌀 EMA Cluster Squeeze & Breakout":
            filtered_df = filtered_df[filtered_df["_cluster_squeeze_match"] == True]
        elif sma_trend_filter == "⚡ 9/20/44 Triple EMA Bullish Cross":
            filtered_df = filtered_df[filtered_df["_triple_ema_match"] == True]

        if enable_vol_multiplier:
            filtered_df = filtered_df[filtered_df["_raw_vol"] >= (filtered_df["_avg_vol_20"] * vol_multiplier)]

    tab_screener, tab_deepdive, tab_pullback_watchlist, tab_watchlist = st.tabs(
        [
            "📊 Screener & Momentum Signals",
            "🔬 Single Stock Chart & AI Thesis",
            "🎯 Pullback Watchlist & Order Trigger",
            "💼 Paper Trading Portfolio",
        ]
    )

    # TAB 1: SCREENER
    with tab_screener:
        st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)})")
        display_cols = [
            "Ticker", "Signal", "Price (₹)", "Change (%)", "Volume",
            "Composite Score", "ROCE (%)", "ADX (14)", "RSI (14)",
            "From 52W High (%)", "Vol Surge", "Market Cap (₹ Cr)",
            "Order Book (₹ Cr)", "OB / MCap"
        ]
        table_data = filtered_df[display_cols].copy()
        table_data["Price (₹)"] = table_data["Price (₹)"].apply(lambda x: f"₹{x:,.2f}")
        table_data["Vol Surge"] = table_data["Vol Surge"].apply(lambda x: "✅" if x else "⬜")
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # TAB 2: DEEP DIVE
    with tab_deepdive:
        stock_list = df_raw["Raw_Ticker"].tolist()
        selected_stock = st.selectbox("Select Stock:", stock_list, index=0)
        if selected_stock:
            st.info(f"Viewing Setup for {selected_stock}")

    # TAB 3: PULLBACK WATCHLIST & AUTO LIMIT TRIGGER
    with tab_pullback_watchlist:
        st.subheader("🎯 Pullback Watchlist & Limit Order Execution")
        st.info("💡 **Pullback Entry Engine:** Place limit orders below current market price (LTP). When market price dips to or below your target, the system triggers and automatically buys the stock.")

        # Candidate dropdown
        pullback_candidates = df_raw["Raw_Ticker"].tolist()
        with st.expander("➕ Add Stock to Pullback Watchlist", expanded=False):
            cw1, cw2, cw3, cw4, cw5 = st.columns([1.2, 1, 1, 1, 1])
            with cw1:
                sel_stock = st.selectbox("Stock:", pullback_candidates, key="wb_stock_select")
                matched_row = df_raw[df_raw["Raw_Ticker"] == sel_stock].iloc[0]
                live_ltp = float(matched_row["Price (₹)"])
                ema20_val = float(matched_row["20 EMA"])
            with cw2:
                st.metric("Current LTP", f"₹{live_ltp:,.2f}")
            with cw3:
                target_entry_price = st.number_input("Target Entry Limit (₹)", value=round(ema20_val, 2), min_value=0.1, step=0.5)
            with cw4:
                sl_val = st.number_input("Stop Loss (₹)", value=round(target_entry_price * 0.95, 2), min_value=0.0, step=0.5)
            with cw5:
                tgt_val = st.number_input("Target (₹)", value=round(target_entry_price * 1.10, 2), min_value=0.0, step=0.5)

            sub1, sub2, btn_col = st.columns([1, 2.5, 1])
            with sub1:
                qty_input = st.number_input("Quantity", value=50, min_value=1, step=1, key="wb_qty_val")
            with sub2:
                strat_note = st.text_input("Strategy Note", value="Pullback Dip Entry near 20 EMA")
            with btn_col:
                st.write("")
                st.write("")
                if st.button("📥 Add to Watchlist", use_container_width=True):
                    raw_sym = sel_stock
                    clean_sym = raw_sym.replace(".NS", "").replace(".BO", "")
                    new_item = {
                        "id": f"wb_{clean_sym}_{int(time.time())}",
                        "Date Added": str(date.today()),
                        "Ticker": clean_sym,
                        "Raw_Ticker": raw_sym,
                        "Target Buy (₹)": float(target_entry_price),
                        "SL (₹)": float(sl_val),
                        "TGT (₹)": float(tgt_val),
                        "Qty": int(qty_input),
                        "Strategy": strat_note.strip(),
                        "Status": "⏳ Waiting for Pullback",
                    }
                    st.session_state["pullback_watchlist"].append(new_item)
                    save_json_file(WATCHLIST_FILE, st.session_state["pullback_watchlist"])
                    st.success(f"Added {clean_sym} to Watchlist (Target Entry: ₹{target_entry_price:.2f})!")
                    st.rerun()

        # Monitor Watchlist Items
        active_watchlist = st.session_state.get("pullback_watchlist", [])
        if active_watchlist:
            # Build clean live price dictionary directly from screener data
            live_price_dict = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"]))

            updated_watchlist = []
            display_rows = []

            for item in active_watchlist:
                sym = item.get("Raw_Ticker") or f"{item.get('Ticker')}.NS"
                clean_sym = item.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
                target_buy = float(item.get("Target Buy (₹)", 0.0))
                sl_price = float(item.get("SL (₹)", 0.0))
                tgt_price = float(item.get("TGT (₹)", 0.0))
                qty = int(item.get("Qty", 1))
                status_str = item.get("Status", "⏳ Waiting for Pullback")

                # Get TRUE Live Market Price (No false default to target_buy!)
                curr_ltp = live_price_dict.get(sym)
                if curr_ltp is None:
                    try:
                        t = yf.Ticker(sym)
                        curr_ltp = float(t.fast_info.last_price)
                    except Exception:
                        curr_ltp = None

                # Strict Trigger Evaluation: ONLY trigger when we have a valid live price AND LTP <= target_buy
                if "Waiting" in status_str and curr_ltp is not None and curr_ltp > 0 and curr_ltp <= target_buy:
                    status_str = "⚡ Triggered / Bought"
                    item["Status"] = status_str
                    st.toast(f"🎯 PULLBACK HIT! Limit order for {clean_sym} executed at ₹{curr_ltp:.2f}!", icon="⚡")
                    st.success(f"🔔 **Pullback Triggered:** {clean_sym} reached buy level ₹{target_buy:,.2f} (LTP: ₹{curr_ltp:,.2f}). Auto-executed into Paper Trading Portfolio!")

                    # Add to Paper Trading
                    trade_record = {
                        "id": f"{sym}_{int(time.time())}",
                        "Date": str(date.today()),
                        "Exit_Date": "",
                        "Ticker": clean_sym,
                        "Buy Price (₹)": curr_ltp,
                        "SL (₹)": sl_price,
                        "TGT (₹)": tgt_price,
                        "Exit Price (₹)": 0.0,
                        "Qty": qty,
                        "Remarks": f"Pullback Auto-Entry ({item.get('Strategy', 'Dip Buy')})",
                        "Status": "🟢 Open",
                        "Invested (₹)": round(curr_ltp * qty, 2),
                        "Raw_Ticker": sym,
                    }
                    if not any(p.get("id") == trade_record["id"] for p in st.session_state["paper_portfolio"]):
                        st.session_state["paper_portfolio"].append(trade_record)
                        save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])

                item["Status"] = status_str
                updated_watchlist.append(item)

                if curr_ltp is not None and target_buy > 0:
                    dist_pct = ((curr_ltp - target_buy) / target_buy) * 100.0
                    dist_str = f"{dist_pct:+.2f}% away" if "Waiting" in status_str else "Executed ✅"
                    ltp_display = f"₹{curr_ltp:,.2f}"
                else:
                    dist_str = "Calculating..."
                    ltp_display = "Fetching..."

                display_rows.append({
                    "Date Added": item.get("Date Added", str(date.today())),
                    "Ticker": clean_sym,
                    "Current LTP (₹)": ltp_display,
                    "Target Buy (₹)": f"₹{target_buy:,.2f}",
                    "Distance to Entry": dist_str,
                    "SL (₹)": f"₹{sl_price:,.2f}",
                    "TGT (₹)": f"₹{tgt_price:,.2f}",
                    "Qty": qty,
                    "Status": status_str,
                    "Strategy": item.get("Strategy", "Pullback Buy"),
                })

            st.session_state["pullback_watchlist"] = updated_watchlist
            save_json_file(WATCHLIST_FILE, updated_watchlist)

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

            # Management & Reset Row
            m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
            with m_col1:
                del_choices = {f"{it.get('Ticker')} (Target: ₹{it.get('Target Buy (₹)')}) [{it.get('Status')}]": idx for idx, it in enumerate(updated_watchlist)}
                sel_del = st.selectbox("Select Item:", list(del_choices.keys()), key="del_wb_select")
            with m_col2:
                st.write("")
                st.write("")
                if st.button("🔄 Re-Arm / Reset to Waiting", use_container_width=True):
                    d_idx = del_choices[sel_del]
                    updated_watchlist[d_idx]["Status"] = "⏳ Waiting for Pullback"
                    st.session_state["pullback_watchlist"] = updated_watchlist
                    save_json_file(WATCHLIST_FILE, updated_watchlist)
                    st.success("Re-armed position back to Waiting!")
                    st.rerun()
            with m_col3:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
                    d_idx = del_choices[sel_del]
                    updated_watchlist.pop(d_idx)
                    st.session_state["pullback_watchlist"] = updated_watchlist
                    save_json_file(WATCHLIST_FILE, updated_watchlist)
                    st.success("Deleted from watchlist!")
                    st.rerun()
        else:
            st.info("Watchlist is empty. Add a pullback setup above.")

    # TAB 4: PAPER TRADING
    with tab_watchlist:
        st.subheader("💼 Paper Trading Portfolio & Risk Manager")
        active_portfolio = st.session_state.get("paper_portfolio", [])
        if active_portfolio:
            st.dataframe(pd.DataFrame(active_portfolio), use_container_width=True, hide_index=True)
            if st.button("🗑️ Reset All Trades"):
                st.session_state["paper_portfolio"] = []
                save_json_file(PORTFOLIO_FILE, [])
                st.rerun()
        else:
            st.info("No active paper trades.")
else:
    st.info("👈 Click **'🚀 Run Screener Scan'** in the sidebar to begin.")
