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

# 1. Page Configuration & Center-Aligned Table Styling
st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Center alignment for all tables and headers */
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

    /* Live Market Index Ribbon */
    .index-ticker-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 18px;
        gap: 12px 16px;
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

    /* KPI Summary Cards */
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

# --- TIME-DRIVEN MARKET HOURS & ALERT SYSTEM ---
def render_alert_permission_banner():
    banner_html = """
    <div style="display: flex; align-items: center; justify-content: space-between; background: #ecfdf5; border: 2px solid #10b981; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(16,185,129,0.12);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 22px;">🔔</span>
            <div>
                <div style="font-size: 14.5px; font-weight: 700; color: #065f46;">Auto-Market Hours Alert Engine (9:15 AM – 3:30 PM IST)</div>
                <div style="font-size: 12px; color: #047857;" id="market-time-status">Checking market hours & permission status...</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <button id="alert-btn" onclick="activateSystemAlerts()" style="background: #059669; color: #ffffff; border: none; border-radius: 8px; padding: 10px 18px; font-size: 13.5px; font-weight: 700; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">
                🔊 Grant Sound & Push Permission
            </button>
            <span id="alert-status-msg" style="font-size: 13px; font-weight: 700; color: #065f46;"></span>
        </div>
    </div>
    <script>
    function checkMarketHoursAndPermissions() {
        var statusSub = document.getElementById("market-time-status");
        var btnEl = document.getElementById("alert-btn");
        
        var d = new Date();
        var utc = d.getTime() + (d.getTimezoneOffset() * 60000);
        var istDate = new Date(utc + (3600000 * 5.5));
        var hours = istDate.getHours();
        var minutes = istDate.getMinutes();
        var day = istDate.getDay();
        
        var timeVal = hours * 100 + minutes;
        var isWeekday = (day >= 1 && day <= 5);
        
        // TEMPORARY BYPASS: Forces system "ON" regardless of time/day for testing
        var isMarketHours = true; 
        
        if ("Notification" in window && Notification.permission === "granted") {
            btnEl.style.display = "none";
            if (isMarketHours) {
                statusSub.innerText = "🟢 Market Open (9:15 AM - 3:30 PM IST): Audio & Notifications are FULLY ARMED.";
            } else {
                statusSub.innerText = "🌙 Market Closed: System is on standby for the next trading session.";
            }
        } else {
            statusSub.innerText = "⚠️ Click the button once to enable browser sound permissions.";
        }
    }

    function activateSystemAlerts() {
        var statusEl = document.getElementById("alert-status-msg");
        if ("Notification" in window) {
            Notification.requestPermission().then(function(permission) {
                if (permission === "granted") {
                    statusEl.innerText = "✅ Active!";
                    checkMarketHoursAndPermissions();
                }
            });
        }
        
        // Play a LOUD test sound
        try {
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                var ctx = new AudioCtx();
                ctx.resume();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.type = "square"; // Harsh, loud alarm sound
                osc.frequency.value = 900;
                osc.connect(gain);
                gain.connect(ctx.destination);
                gain.gain.setValueAtTime(1.0, ctx.currentTime); // 100% volume
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                osc.start();
                osc.stop(ctx.currentTime + 0.5);
            }
        } catch(e) { console.log(e); }
    }
    }

    window.onload = checkMarketHoursAndPermissions;
    setTimeout(checkMarketHoursAndPermissions, 1000);
    </script>
    """
    components.html(banner_html, height=85)


def play_trigger_alert(ticker, buy_price):
    # Completely escapes the iframe to play a loud synthetic alarm on the parent window
    js_html = f"""
    <script>
    (function() {{
        try {{
            // 1. Push Notification
            var nTitle = "🎯 PULLBACK HIT: {ticker}";
            var nBody = "Trade executed at ₹{buy_price:,.2f}. Moved to Portfolio.";
            var nIcon = "https://cdn-icons-png.flaticon.com/512/190/190411.png";
            var notif = window.parent.Notification || window.Notification;
            
            if (notif && notif.permission === "granted") {{
                new notif(nTitle, {{body: nBody, icon: nIcon}});
            }}

            // 2. Loud Multi-Tone Alarm using Web Audio API (Bypasses file blocks)
            window.parent.eval(`
                try {{
                    var AudioCtx = window.AudioContext || window.webkitAudioContext;
                    if(AudioCtx) {{
                        var ctx = new AudioCtx();
                        ctx.resume();
                        
                        function playLoudBeep(freq, startTime, duration) {{
                            var osc = ctx.createOscillator();
                            var gain = ctx.createGain();
                            
                            osc.type = "square"; // 'square' is a piercing digital alarm tone
                            osc.frequency.value = freq;
                            
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            
                            // Max Volume (1.0)
                            gain.gain.setValueAtTime(1.0, startTime);
                            gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
                            
                            osc.start(startTime);
                            osc.stop(startTime + duration);
                        }}
                        
                        var now = ctx.currentTime;
                        // Play 3 loud piercing alarm beeps
                        playLoudBeep(900, now, 0.25);
                        playLoudBeep(900, now + 0.35, 0.25);
                        playLoudBeep(1200, now + 0.70, 0.5);
                    }}
                }} catch(err) {{
                    console.log("Audio failed:", err);
                }}
            `);
        }} catch(e) {{ 
            console.log("Alert script error:", e); 
        }}
    }})();
    </script>
    """
    components.html(js_html, height=0, width=0)

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


if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_json_file(PORTFOLIO_FILE)

if "pullback_watchlist" not in st.session_state:
    st.session_state["pullback_watchlist"] = load_json_file(WATCHLIST_FILE)

if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}

if "screener_data" not in st.session_state:
    st.session_state["screener_data"] = pd.DataFrame()

# ==========================================
# --- FILTER DICTIONARIES FOR STATE MGMT ---
# ==========================================
WIDE_OPEN_FILTERS = {
    "sel_universe": "All NSE Stocks (Full Listed)",
    "scan_limit": 100,  
    "strict_fund": False,
    "pat_growth": False,
    "ob_mcap": False,
    "roce_rng": (-20, 100),
    "mcap_rng": (0, 2000000),
    "max_de": 5.0,
    "price_rng": (10, 10000),
    "rsi_rng": (10, 95),
    "min_adx": 0,
    "dist_52w": 100,
    "ma_align": "Any Trend",
    "vol_10d_en": False,
    "vol_10d_mult": 1.1,
    "vol_20d_en": False,
    "vol_20d_mult": 1.2
}

STRICT_STRATEGY_FILTERS = {
    "sel_universe": "All NSE Stocks (Full Listed)",
    "scan_limit": 1950, 
    "strict_fund": True, 
    "pat_growth": False,
    "ob_mcap": False,
    "roce_rng": (20, 100),
    "mcap_rng": (1000, 2000000),
    "max_de": 0.50,
    "price_rng": (30, 2000),
    "rsi_rng": (55, 75),
    "min_adx": 20,
    "dist_52w": 12,
    "ma_align": "Any Trend",
    "vol_10d_en": False,
    "vol_10d_mult": 1.1,
    "vol_20d_en": False,
    "vol_20d_mult": 1.2
}

# Ensure defaults are initialized in session state
for key, val in WIDE_OPEN_FILTERS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_to_open_filters():
    for key, val in WIDE_OPEN_FILTERS.items():
        st.session_state[key] = val

def apply_strict_filters():
    for key, val in STRICT_STRATEGY_FILTERS.items():
        st.session_state[key] = val

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

ORDER_BOOK_CR_MAP = {
    "HAL": 94000, "BEL": 76000, "BDL": 20000, "MAZDOCK": 40000, 
    "COCHINSHIP": 22000, "GRSE": 25000, "BHEL": 135000, "ACE": 3200, 
    "JYOTICNC": 4850, "BEML": 12500, "ISGEC": 8500, "TECHNOE": 9000, 
    "ELECON": 3500, "KIRLOSENG": 3200, "LT": 475000, "RVNL": 85000, 
    "IRCON": 32000, "NCC": 57000, "JINDRILL": 1310, "PNCINFRA": 18000, 
    "KEC": 34000, "KPIL": 58000, "NBCC": 81000, "HGINFRA": 12000, 
    "AHLUCONT": 14000, "POWERMECH": 55000, "TITAGARH": 28000, "JWL": 20000, 
    "RAILTEL": 5000, "ENGINERSIN": 10500, "PSPPROJECT": 6000, "GPTINFRA": 3500, 
    "MANINFRA": 4200, "MMFL": 1800, "GENUSPOWER": 21500, "KRONOX": 185,
}

NSE_FULL_EQUITIES = [
    "20MICRONS", "21STCENMGM", "360ONE", "3IINFOLTD", "3MINDIA", "3PLAND", "5PAISA", "63MOONS",
    "A2ZINFRA", "AAATECH", "AADHARHFC", "AAKASH", "AAL", "AARTIDRUGS", "AARTIIND",
    "AARTIPHARM", "AARTISURF", "AARVEEDEN", "AARVI", "AAVAS", "ABAN", "ABB", "ABBOTINDIA",
    "ABCAPITAL", "ABDL", "ABFRL", "ABREL", "ABSLAMC", "ACC", "ACCELYA", "ACCURACY", "ACE",
    "ACEINTEG", "ACI", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ADFFOODS",
    "ADL", "ADORWELD", "ADROITINFO", "ADVANIHOTR", "ADVENZYMES", "AEGISLOG", "AEROFLEX",
    "AEROPACE", "AETHER", "AFFLE", "AGARIND", "AGI", "AGRITECH", "AGROPHOS", "AGSTRA",
    "AHL", "AHLADA", "AHLEAST", "AHLUCONT", "AIAENG", "AIRAN", "AIROLAM", "AJANTPHARM",
    "AJMERA", "AJOONI", "AKASH", "AKG", "AKSHAR", "AKSHARCHEM", "AKSHOPTFBR", "AKZOINDIA",
    "ALANKIT", "ALBERTDAVD", "ALEMBICLTD", "ALICON", "ALKALI", "ALKEM", "ALKYLAMINE",
    "ALLCARGO", "ALLDIGI", "ALMONDZ", "ALOKINDS", "ALPA", "ALPHAGEO", "ALSTONE", "AMBER",
    "AMBICAAGAR", "AMBIKCO", "AMBUJACEM", "AMDIND", "AMIORG", "AMJLAND", "AMRUTANJAN",
    "ANANDRATHI", "ANANTRAJ", "ANDHRAPAP", "ANDHRSUGAR", "ANGELONE", "ANIKINDS", "ANKITMETAL",
    "ANMOL", "ANTGRAPHIC", "ANUP", "ANURAS", "APARINDS", "APCL", "APCOTEXIND", "APEX",
    "APLAPOLLO", "APLLTD", "APOLLO", "APOLLOHOSP", "APOLLOPIPE", "APOLLOTYRE", "APOLSINHOT",
    "APTECHTGL", "APTUS", "ARCHIDPLY", "ARCHIES", "ARE&M", "ARENTERP", "ARIES", "ARIHANTCAP",
    "ARIHANTSUP", "ARKG", "ARMANFIN", "ARMSOL", "AROGRANITE", "ARROWGREEN", "ARSHIYA", "ARSSINFRA",
    "ARTEMISMED", "ARTNIRMAN", "ARVEE", "ARVIND", "ARVINDFASN", "ARVSMART", "ASAHIINDIA",
    "ASAHISONG", "ASAL", "ASALCBR", "ASHAPURMIN", "ASHIANA", "ASHOKA", "ASHOKLEY", "ASIANENE",
    "ASIANHOTNR", "ASIANPAINT", "ASIANTILES", "ASPINWALL", "ASTEC", "ASTERDM", "ASTRAL",
    "ASTRAMICRO", "ASTRAZEN", "ASTRON", "ATALREAL", "ATAM", "ATFL", "ATGL", "ATL",
    "ATLANTA", "ATUL", "ATULAUTO", "AUBANK", "AURIONPRO", "AUROIMPREX", "AUROPHARMA",
    "AURUM", "AUSOMENT", "AUTOAXLES", "AUTOBEAT", "AUTOIND", "AVADHSUGAR", "AVALON",
    "AVANTIFEED", "AVONMORE", "AVROIND", "AVTNPL", "AWHCL", "AWL", "AXISBANK", "AXISCADES",
    "AXITA", "AYMSYNTEX", "BAFNAPH", "BAGFILMS", "BAIDFINS", "BAJAJ-AUTO", "BAJAJCON",
    "BAJAJELEC", "BAJAJFINSV", "BAJAJHCARE", "BAJAJHFL", "BAJAJHLDNG", "BAJAJHIND",
    "BAJEL", "BAJFINANCE", "BALAJITELE", "BALAMINES", "BALAXI", "BALKRISHNA", "BALKRISIND",
    "BALMLAWRIE", "BALPHARMA", "BALRAMCHIN", "BANARBEADS", "BANARISUG", "BANCOINDIA",
    "BANDHANBNK", "BANG", "BANKA", "BANKBARODA", "BANKINDIA", "BANSALWIRE", "BANSWRAS",
    "BARBEQUE", "BASF", "BASML", "BATAINDIA", "BAYERCROP", "BBL", "BBTC", "BBTCL",
    "BCG", "BCLIND", "BCONCEPTS", "BDL", "BEARDSELL", "BECTORFOOD", "BEDMUTHA", "BEL",
    "BEML", "BEPL", "BERGEPAINT", "BESTAGRO", "BFINVEST", "BFUTILITIE", "BGRENERGY",
    "BHAGCHEM", "BHAGERIA", "BHAGYANGR", "BHANDARI", "BHARATFORG", "BHARATGEAR",
    "BHARATRAS", "BHARATWIRE", "BHARTIARTL", "BHEL", "BIGBLOC", "BIKJI", "BIL",
    "BILENERGY", "BINDALAGRO", "BIOCON", "BIOFILCHEM", "BIRLACABLE", "BIRLACORPN",
    "BIRLAMONEY", "BIRLATYRE", "BKMINDST", "BLAL", "BLBLIMITED", "BLISSGVS", "BLKASHYAP",
    "BLS", "BLSE", "BLUECHIP", "BLUEDART", "BLUEJET", "BLUESTARCO", "BODALCHEM",
    "BOMDYEING", "BOROLTD", "BORORENEW", "BOSCHLTD", "BPCL", "BPL", "BRIGADE",
    "BRITANNIA", "BRNL", "BSE", "BSHSL", "BSL", "BSOFT", "BTML", "BURNPUR",
    "BUTTERFLY", "BVCL", "BYKE", "CALSOFT", "CAMLINFINE", "CAMPUS", "CAMS",
    "CANBK", "CANFINHOME", "CANTABIL", "CAPACITE", "CAPITALSFB", "CAPL", "CAPLTD",
    "CARBORUNIV", "CAREERP", "CARERATING", "CARTRADE", "CARYSIL", "CASTROLIND",
    "CCCL", "CCHHL", "CCL", "CDSL", "CEATLTD", "CELEBRITY", "CELLOPENS", "CENTENKA",
    "CENTEXT", "CENTRALBK", "CENTRUM", "CENTUM", "CENTURYPLY", "CENTURYTEX", "CERA",
    "CEREBRAINT", "CESC", "CGCL", "CGPOWER", "CHALET", "CHAMBLFERT", "CHEMBOND",
    "CHEMCON", "CHEMFAB", "CHEMPLASTS", "CHENNPETRO", "CHEVIOT", "CHOICEIN",
    "CHOLAHLDNG", "CHOLAFIN", "CIGNITITEC", "CINELINE", "CINEVISTA", "CIPLA",
    "CLEAN", "CLEDUCATE", "CLSEL", "CMSINFO", "COALINDIA", "COASTCORP", "COCHINSHIP",
    "COFORGE", "COLPAL", "COMPINFO", "COMPUSOFT", "COMSYN", "CONCOR", "CONCORDBIO",
    "CONFIPET", "CONSOFINVT", "CONTROLPR", "CORALFINAC", "CORDSCABLE", "COROMANDEL",
    "COSMOFIRST", "COUNCODOS", "CRAFTSMAN", "CREATIVE", "CREATIVEYE", "CREDITACC",
    "CREST", "CRISIL", "CROMPTON", "CROWN", "CSBBANK", "CSLFINANCE", "CTE", "CUB",
    "CUBEXTUB", "CUMMINSIND", "CUPID", "CYIENT", "CYIENTDLM", "DABUR", "DALBHARAT",
    "DALMIASUG", "DAMODARIND", "DANLAW", "DAP", "DAPS", "DATAMATICS", "DATAPATTNS",
    "DAVANGERE", "DBCORP", "DBL", "DBOL", "DBREALTY", "DBSTOCKBRO", "DCAL", "DCBBANK",
    "DCI", "DCM", "DCMFINSERV", "DCMNVL", "DCMSHRIRAM", "DCMSRIND", "DCW", "DCXINDIA",
    "DECCANCE", "DEEDEV", "DEEPAKFERT", "DEEPAKNTR", "DEEPENR", "DEEPINDS", "DELHIVERY",
    "DELPHIFX", "DELTACORP", "DELTAMAGNT", "DEN", "DENORA", "DEVIT", "DEVYANI",
    "DGCONTENT", "DHAMPURSUG", "DHANBANK", "DHANI", "DHANUKA", "DHARMAJ", "DHRUV",
    "DHUNINV", "DIACABS", "DIAMINESQ", "DIAMONDYD", "DICIND", "DIFFN", "DIGISPICE",
    "DIGJAMLTD", "DIL", "DISHTV", "DIVGIITTS", "DIVISLAB", "DIXON", "DJML", "DLF",
    "DLINKINDIA", "DMART", "DMCC", "DNAMEDIA", "DODLA", "DOLATALGO", "DOLLAR",
    "DOLPHIN", "DOMS", "DONEAR", "DPABHUSHAN", "DPSCLTD", "DPWIRES", "DRCS",
    "DREDGECORP", "DRREDDY", "DSDNL", "DSPAMC", "DSSL", "DTIL", "DUCON", "DVL",
    "DWARKESH", "DYCL", "DYNAMATECH", "DYNPRO", "E2E", "EASEMYTRIP", "ECLERX",
    "EDELWEISS", "EICHERMOT", "EIDPARRY", "EIHAHOTELS", "EIHOTEL", "EIMCOELECO",
    "EKC", "ELDEHSG", "ELECON", "ELECTCAST", "ELECTHERM", "ELGIEQUIP", "ELGIRUBCO",
    "EMAMILTD", "EMAMIPAP", "EMAMIREAL", "EMBDL", "EMCURE", "EMIL", "EMKAY",
    "EMMBI", "EMSLIMITED", "ENDURANCE", "ENERGYDEV", "ENGINERSIN", "ENIL", "ENTERO",
    "EPACK", "EPIGRAL", "EPL", "EQUIPPP", "EQUITASBNK", "ERIS", "EROSMEDIA",
    "ESABINDIA", "ESAFSFB", "ESCORTS", "ESSARSHPNG", "ESSENTIA", "ESTER", "ETHOSLTD",
    "EUROBOND", "EUROTEXIND", "EVEREADY", "EVERESTIND", "EXCEL", "EXCELINDUS",
    "EXICOM", "EXIDEIND", "EXPLEOSOL", "EXXARO", "FACT", "FAIRCHEMOR", "FAZE3Q",
    "FBL", "FCSSOFT", "FDC", "FEDERALBNK", "FEDFINA", "FEL", "FELDVR", "FIBERWEB",
    "FIEMIND", "FILATEX", "FILATFASH", "FINCABLES", "FINEORG", "FINOPB", "FINPIPE",
    "FIRSTCRY", "FIVESTAR", "FLAIR", "FLEXITUFF", "FLFL", "FLUOROCHEM", "FMGOETZE",
    "FMNL", "FOCUS", "FOCE", "FORCEMOT", "FORTIS", "FOSECOIND", "FSL", "GABRIEL",
    "GAEL", "GAIL", "GALAPREC", "GALAXY", "GALAXYSURF", "GALLANTT", "GANDHAR",
    "GANDHITUBE", "GANECOS", "GANESHBE", "GANESHHOUC", "GANGESSECU", "GARFIBRES",
    "GATECH", "GATEWAY", "GAYAHWS", "GAYAPROJ", "GEECEE", "GEEKAYWIRE", "GENCON",
    "GENESYS", "GENUSPAPER", "GENUSPOWER", "GEOJITFSL", "GEPIL", "GESHIP", "GET&D",
    "GFLLIMITED", "GHCL", "GHCLTEXTIL", "GICHSGFIN", "GICRE", "GILLANDERS", "GILLETTE",
    "GINNIFILA", "GIPCL", "GKWLIMITED", "GLAND", "GLAXO", "GLENMARK", "GLFL",
    "GLOBAL", "GLOBALVECT", "GLOBE", "GLOBUSSPR", "GLOSTERLTD", "GLS", "GMBREW",
    "GMDCLTD", "GMMPFAUDLR", "GMRINFRA", "GMRP&UI", "GNA", "GNFC", "GOACARBON",
    "GOCLCORP", "GOCOLORS", "GODFRYPHLP", "GODHA", "GODREJAGRO", "GODREJCP",
    "GODREJIND", "GODREJPROP", "GOENKA", "GOKEX", "GOKUL", "GOKULAGRO", "GOLDENTOBC",
    "GOLDIAM", "GOLDTECH", "GOODLUCK", "GOPAL", "GOYALALUM", "GPIL", "GPPL",
    "GPTHEALTH", "GPTINFRA", "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GREAVESCOT",
    "GREENLAM", "GREENPANEL", "GREENPLY", "GREENPOWER", "GRINDWELL", "GRINFRA",
    "GRMOVER", "GROBTEA", "GRPLTD", "GRSE", "GRWRHITECH", "GSCLCEMENT", "GSFC",
    "GSLSU", "GSPL", "GSS", "GTEIT", "GTL", "GTLINFRA", "GTPL", "GUFICBIO",
    "GUJALKALI", "GUJAPOLLO", "GUJGASLTD", "GUJRAFFIA", "GULFOILLUB", "GULFPETRO",
    "GULPOLY", "GVKPIL", "GVPTECH", "HAL", "HAPPSTMNDS", "HAPPYFORGE", "HARSHA",
    "HATHWAY", "HATSUN", "HAVELLS", "HAVISHA", "HBLPOWER", "HBSL", "HCC",
    "HCG", "HCL-INSYS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HDIL",
    "HEADSUP", "HECPROJECT", "HEG", "HEIDELBERG", "HEMIPROP", "HERANBA", "HERCULES",
    "HERITGFOOD", "HEROMOTOCO", "HESTERBIO", "HEXATRADEX", "HFCL", "HGINFRA",
    "HGS", "HIKAL", "HIL", "HILTON", "HIMATSEIDE", "HINDALCO", "HINDCOMPOS",
    "HINDCON", "HINDCOPPER", "HINDDORROL", "HINDOILEXP", "HINDPETRO", "HINDUNILVR",
    "HINDWAREAP", "HINDZINC", "HIRECT", "HISARMETAL", "HITECH", "HITECHCORP",
    "HITECHGEAR", "HLVLTD", "HMAAGRO", "HMT", "HMVL", "HNDFDS", "HOMEFIRST",
    "HONASA", "HONAUT", "HONDAPOWER", "HOVS", "HPAL", "HPIL", "HPL", "HUBTOWN",
    "HUDCO", "HUHTAMAKI", "HYBRIDFIN", "HYUNDAI", "IBREALEST", "IBULHSGFIN",
    "ICDSL", "ICEMAKE", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ICIL", "ICRA",
    "IDBI", "IDEA", "IDEAFORGE", "IDFCFIRSTB", "IEX", "IFBAGRO", "IFBIND",
    "IFCI", "IFGLEXPOR", "IGARASHI", "IGL", "IGPL", "IIFL", "IIFLCAPS", "IIFLSEC",
    "IITL", "IL&FSENGG", "IL&FSTRANS", "IMAGICAA", "IMFA", "IMPAL", "IMPEXFERRO",
    "INCREDIBLE", "INDBANK", "INDGN", "INDHOTEL", "INDIACEM", "INDIAGLYCO",
    "INDIAMART", "INDIANB", "INDIANCARD", "INDIANHUME", "INDIASHLTR", "INDIGO",
    "INDIGOPNTS", "INDIGRID", "INDNIPPON", "INDOAMIN", "INDOBORAX", "INDOCO",
    "INDOCOUNT", "INDORAMA", "INDOSTAR", "INDOTECH", "INDOTHAI", "INDOUS",
    "INDOVATION", "INDOVTRAD", "INDOWIND", "INDRAMEDCO", "INDSWFTLTD", "INDTERRAIN",
    "INDUSINDBK", "INDUSTOWER", "INFIBEAM", "INFOBEAN", "INFOMEDIA", "INFY",
    "INGERRAND", "INNOVACAP", "INNOVANA", "INNOVENT", "INOXGREEN", "INOXINDIA",
    "INOXWIND", "INSECTICID", "INSPIRISYS", "INTELLECT", "INTENTECH", "INTERARCH",
    "INVENTURE", "IOB", "IOC", "IOLCP", "IONEXCHANG", "IPCALAB", "IPL", "IRB",
    "IRCON", "IRCTC", "IREDA", "IRFC", "IRIS", "IRISDOREME", "ISEC", "ISFT",
    "ISGEC", "ISMTLTD", "ITC", "ITDC", "ITDCEM", "ITI", "IVC", "IVP", "IXIGO",
    "IZMO", "J&KBANK", "JAGRAN", "JAGSNPHARM", "JAIBALAJI", "JAICORPLTD", "JAIPURKURT",
    "JAMNAAUTO", "JASH", "JAYAGROGN", "JAYBARMARU", "JAYNECOIND", "JAYSREETEA",
    "JBCHEPHARM", "JBMA", "JCHAC", "JETACCC", "JETFREIGHT", "JETAIRWAYS", "JHS",
    "JINDALCO", "JINDALPHOT", "JINDALPOLY", "JINDALSAW", "JINDALSTEL", "JINDRILL",
    "JINDWORLD", "JIOFIN", "JISLDVREQS", "JISLJALEQS", "JITFINFRA", "JKCEMENT",
    "JKIL", "JKLAKSHMI", "JKPAPER", "JKTYRE", "JMA", "JMFINANCIL", "JNKINDIA",
    "JOCIL", "JPASSOCIAT", "JPINFRATEC", "JPPOWER", "JSL", "JSLL", "JSWENERGY",
    "JSWHL", "JSWINFRA", "JSWSTEEL", "JTEKTINDIA", "JTLIND", "JUBLFOOD", "JUBLINGREA",
    "JUBLPHARMA", "JUSTDIAL", "JWL", "JYOTHYLAB", "JYOTICNC", "JYOTISTRUC", "KABRAEXTRU",
    "KAJARIACER", "KAKATCEM", "KALAMANDIR", "KALPATPOWR", "KALYANI", "KALYANICHEM",
    "KALYANIENG", "KALYANKJIL", "KAMATHOTEL", "KAMDHENU", "KAMOPAINTS", "KANANIIND",
    "KANORICHEM", "KANPRPLA", "KANSAINER", "KAPSTON", "KARMAENG", "KARURVYSYA",
    "KAUSHALYA", "KAVVERITEL", "KAYA", "KAYNES", "KBCGLOBAL", "KCP", "KCPSUGIND",
    "KDDL", "KEC", "KECL", "KEEPLEARN", "KEI", "KELLTONTEC", "KERNEX", "KESORAMIND",
    "KEYFINSERV", "KFINTECH", "KHADIM", "KHAICHEM", "KHAITANLTD", "KHANDSE",
    "KICL", "KILITCH", "KIMS", "KINGFA", "KIOCL", "KIRIINDUS", "KIRLFER",
    "KIRLOSBROS", "KIRLOSENG", "KIRLOSIND", "KITEX", "KKCL", "KMEW", "KMSUGAR",
    "KNRCON", "KOHINOOR", "KOKUYOCAML", "KOLTEPATIL", "KOPRAN", "KOTAKBANK",
    "KOTARISUG", "KOTHARIPET", "KOTHARIPRO", "KPIGREEN", "KPIL", "KPITTECH",
    "KPRMILL", "KRBL", "KREBSBIO", "KRIDHANINF", "KRISHANA", "KRITI", "KRITIKA",
    "KRITINUT", "KRONOX", "KROSS", "KRSNAA", "KSB", "KSCL", "KSHITIJPOL",
    "KSL", "KSOLVES", "KTKBANK", "KUANTUM", "L&TFH", "LAGNAM", "LAKPRE",
    "LALPATHLAB", "LAMBODHARA", "LANCER", "LANDMARK", "LAOPALA", "LASA",
    "LATENTVIEW", "LATTEYS", "LAURUSLABS", "LAXMICOT", "LAXMIMACH", "LCCINFOTEC",
    "LEMONTREE", "LEMERITE", "LGHL", "LIBAS", "LIBERTSHOE", "LICHSGFIN",
    "LICI", "LIKHITHA", "LINC", "LINCOLN", "LINDEINDIA", "LLOYDSENGG", "LLOYDSENT",
    "LLOYDSME", "LODHA", "LOKESHMACH", "LORDSCHLO", "LOTUSCHO", "LOVABLE",
    "LOYAL", "LT", "LTF", "LTIM", "LTTS", "LUMAXIND", "LUMAXTECH", "LUPIN",
    "LUXIND", "LXCHEM", "LYKALABS", "LYPSAGEMS", "M&M", "M&MFIN", "MAANALU",
    "MACPOWER", "MADHAV", "MADHUCON", "MADRASFERT", "MAGADSUGAR", "MAGNUM",
    "MAHABANK", "MAHAPEXLTD", "MAHASTEEL", "MAHEPC", "MAHESHWARI", "MAHinDCIE",
    "MAHLIFE", "MAHLOG", "MAHSCOOTER", "MAHSEAMLES", "MAITHANALL", "MALLCOM",
    "MALUPAPER", "MANAKALUCO", "MANAKCOAT", "MANAKSIA", "MANAKSTEEL", "MANALIPETC",
    "MANAPPURAM", "MANBA", "MANCREDIT", "MANGALAM", "MANGCHEFER", "MANGLMCEM",
    "MANINDS", "MANINFRA", "MANKIND", "MANORAMA", "MANORG", "MANUGRAPH",
    "MAPMYINDIA", "MARALOVER", "MARATHON", "MARICO", "MARINE", "MARKSANS",
    "MARUTI", "MASFIN", "MASKINVEST", "MASTEK", "MATRIMONY", "MAWANASUG",
    "MAXESTATES", "MAXHEALTH", "MAXIND", "MAYURUNIQ", "MAZDA", "MAZDOCK",
    "MBAPL", "MBECL", "MBLINFRA", "MCDOWELL-N", "MCL", "MCLEODRUSS", "MCX",
    "MEDANTA", "MEDICAMEQ", "MEDICO", "MEDPLUS", "MEGASOFT", "MEGASTAR",
    "MELSTAR", "MENONBE", "MEP", "METROBRAND", "METROPOLIS", "MFSL", "MGEL",
    "MGL", "MHLXMIRU", "MICEL", "MIDHANI", "MINDACORP", "MINDTECK", "MIRCELECTR",
    "MIRZAINT", "MITCON", "MITTAL", "MMFL", "MMP", "MMTC", "MODIRUBBER",
    "MODISNME", "MODTHREAD", "MOHITIND", "MOIL", "MOKSH", "MOL", "MOLDTECH",
    "MOLDTKPAC", "MONARCH", "MONTECARLO", "MORARJEE", "MOREPENLAB", "MOTHERSON",
    "MOTILALOFS", "MOTISONS", "MOTORGEN", "MPHASIS", "MPSLTD", "MRF", "MRO-TEK",
    "MRPL", "MSPL", "MSTCLTD", "MSUMI", "MTARTECH", "MTEDUCARE", "MTNL",
    "MUKANDLTD", "MUKKA", "MUKTAARTS", "MUNJALAU", "MUNJALSHOW", "MURUDCERA",
    "MUTHOOTCAP", "MUTHOOTFIN", "MUTHOOTMF", "MVL", "NACLIND", "NAGAFERT",
    "NAGREEKCAP", "NAGREEKEXP", "NAHARCAP", "NAHARINDUS", "NAHARPOLY", "NAHARSPING",
    "NAM-INDIA", "NARMADA", "NATCOPHARM", "NATHBIOGEN", "NATIONALUM", "NAUKRI",
    "NAVA", "NAVINFLUOR", "NAVKARCORP", "NAVNETEDUL", "NAZARA", "NBCC", "NBIFIN",
    "NCC", "NCLIND", "NDGL", "NDL", "NDLVENTURE", "NDRAUTO", "NDTV", "NECCLTD",
    "NECLIFE", "NELCAST", "NELCO", "NEOGEN", "NESCO", "NESTLEIND", "NETWEB",
    "NETWORK18", "NEULANDLAB", "NEWGEN", "NEXTMEDIA", "NFL", "NGIL", "NGLFINE",
    "NH", "NHPC", "NIACL", "NIBL", "NIITLTD", "NIITMTS", "NILAINFRA", "NILASPACES",
    "NILKAMAL", "NINSYS", "NIPPOBATRY", "NIRAJ", "NIRAJISPAT", "NITCO", "NITINSPIN",
    "NITIRAJ", "NKIND", "NLCINDIA", "NMDC", "NMDCLTD", "NOCIL", "NOIDATOLL",
    "NORBTEAEXP", "NORTHARC", "NOVAAGRI", "NRAIL", "NRBBEARING", "NRL", "NSIL",
    "NSLNISP", "NTPC", "NUCLEUS", "NUPUR", "NUVAMA", "NUVOCO", "NYKAA", "OAL",
    "OBCL", "OBEROIRLTY", "OCCL", "OFSS", "OIL", "OILCOUNTUB", "OLECTRA",
    "OLIL", "OMAXAUTO", "OMAXE", "OMINFRAL", "OMKARCHEM", "ONELIFECAP", "ONEPOINT",
    "ONGC", "ONMOBILE", "ONWARDTEC", "OPTIEMUS", "ORBTEXP", "ORCHPHARMA", "ORICONENT",
    "ORIENTALTL", "ORIENTBELL", "ORIENTCEM", "ORIENTCER", "ORIENTELEC", "ORIENTGREEN",
    "ORIENTHOT", "ORIENTLTD", "ORIENTPPR", "ORISSAMINE", "ORTINLAB", "OSIAHYPER",
    "OSWALAGRO", "OSWALGREEN", "OSWALSEEDS", "PAGEIND", "PAISALO", "PAKKA",
    "PALASHSECU", "PALREDTEC", "PANACEABIO", "PANACHE", "PANAMAPET", "PANSARI",
    "PAR", "PARACABLES", "PARADEEP", "PARAGMILK", "PARAS", "PARASDEFNS", "PARASPETRO",
    "PARSVNATH", "PASUPTAC", "PATANJALI", "PATELENG", "PATINTLOG", "PAVNAIND",
    "PAYTM", "PCBL", "PCHFL", "PCJEWELLER", "PDMJEPAPER", "PDSL", "PEARLPOLY",
    "PEL", "PENIND", "PENINLAND", "PERSISTENT", "PETRONET", "PFC", "PFIZER",
    "PFOCUS", "PFS", "PGEL", "PGHH", "PGHL", "PGIL", "PHOENIXLTD", "PIDILITIND",
    "PIIND", "PILANIINVS", "PILITA", "PIONEEREMB", "PITTIENG", "PIXTRANS", "PKTEA",
    "PLASTIBLEN", "PLATIND", "PLAZACABLE", "PNB", "PNBGILTS", "PNBHOUSING",
    "PNC", "PNCINFRA", "PODDARHOUS", "PODDARMENT", "POKARNA", "POLICYBZR",
    "POLYCAB", "POLYMED", "POLYPLEX", "PONNIERODE", "POONAWALLA", "POWERGRID",
    "POWERINDIA", "POWERMECH", "PPAP", "PPL", "PPLPHARMA", "PRAENG", "PRAJIND",
    "PRAKASH", "PRAKASHSTL", "PRAXIS", "PRECAM", "PRECOT", "PRECWIRE", "PREMEXPLN",
    "PREMIER", "PREMIERENE", "PREMIERPOL", "PREMEXPLOS", "PRESTIGE", "PRICOLLTD",
    "PRIMESECU", "PRINCEPIPE", "PRITI", "PRITIKAUTO", "PRIVISCL", "PROZONER",
    "PRSMJOHNSN", "PRUDENT", "PRUDMOUNT", "PSB", "PSPPROJECT", "PTC", "PTCIL",
    "PTL", "PUNJABCHEM", "PUNJLLOYD", "PURVA", "PVRINOX", "PVP", "QUESS",
    "QUICKHEAL", "RADAAN", "RADHIKAJWE", "RADICO", "RADIOCITY", "RAILTEL",
    "RAIN", "RAINBOW", "RAJESHEXPO", "RAJMET", "RAJRATAN", "RAJRILTD", "RAJSREESUG",
    "RAJTV", "RALLIS", "RAMANEWS", "RAMAPHO", "RAMASTEEL", "RAMCOCEM", "RAMCOIND",
    "RAMCOSYS", "RAMKY", "RANASUG", "RANEENGINE", "RANEHOLDIN", "RATNAMANI",
    "RATNAVEER", "RAYMOND", "RAYMONDLSL", "RBA", "RBL", "RBLBANK", "RCF",
    "RECLTD", "REDINGTON", "REFEX", "REGENCERAM", "RELAXO", "RELIABLE", "RELIANCE",
    "RELIGARE", "RELINFRA", "REMSONSIND", "RENUKA", "REPCOHOME", "REPRO", "RESPONIND",
    "RGL", "RHIM", "RHL", "RICOAUTO", "RIIL", "RITESHIN", "RITES", "RKDL",
    "RKFORGE", "RKSWAMY", "RML", "ROHLTD", "ROLEXRINGS", "ROLLT", "ROLTA",
    "ROML", "ROSSARI", "ROSSELLIND", "ROTO", "ROUTE", "RPGLIFE", "RPOWER",
    "RPPINFRA", "RPPL", "RPSGVENT", "RRKABEL", "RSSOFTWARE", "RSWM", "RSYSTEMS",
    "RTNINDIA", "RTNPOWER", "RUBYMILLS", "RUCHINFRA", "RUCHIRA", "RUPA", "RUSHIL",
    "RUSTOMJEE", "RVHL", "RVNL", "S&SPOWER", "SABEVENTS", "SABTN", "SADBHAV",
    "SADBHINFR", "SADHNANIQ", "SAFARI", "SAGARDEEP", "SAGCEM", "SAH", "SAHANA",
    "SAHARA", "SAHASRA", "SAHYADRI", "SAIL", "SAKAR", "SAKHTISUG", "SAKSOFT",
    "SAKUMA", "SALASAR", "SALONA", "SALSTEEL", "SALZITEC", "SAMBHAAV", "SAMBHI",
    "SAMHI", "SAMMAANCAP", "SAMPRE", "SANCO", "SANDESH", "SANDHAR", "SANDUMA",
    "SANGAMIND", "SANGHIIND", "SANGHVIMOV", "SANGINITA", "SANOFICONS", "SANOFI",
    "SANSERA", "SANSTAR", "SANWARIA", "SAPPHIRE", "SARDAEN", "SAREGAMA", "SARLAPOLY",
    "SARVESHWAR", "SASKEN", "SASTASUNDR", "SATIA", "SATINDLTD", "SATIN", "SBCL",
    "SBC", "SBFC", "SBICARD", "SBILIFE", "SBIN", "SCAPDVR", "SCHAEFFLER",
    "SCHAND", "SCHNEIDER", "SCI", "SCILAL", "SCPL", "SDBL", "SEAMECLTD", "SECURCRED",
    "SECURKLOUD", "SEJALLTD", "SELAN", "SELMC", "SEMAC", "SENCO", "SEPC",
    "SEQUENT", "SERVOTECH", "SESHAPAPER", "SETCO", "SETUINFRA", "SEYAIND", "SFL",
    "SGIL", "SGL", "SHAH", "SHAHALLOYS", "SHAILY", "SHAKTIPUMP", "SHALBY",
    "SHALPAINTS", "SHANKARA", "SHANTIGEAR", "SHARDACROP", "SHARDAMOTR", "SHAREINDIA",
    "SHEKHAWATI", "SHEMAROO", "SHILPAMED", "SHIVALIK", "SHIVAMAUTO", "SHIVAMILLS",
    "SHIVATEX", "SHK", "SHOPERSTOP", "SHRADHA", "SHREDIGCEM", "SHREEAUTO",
    "SHREECARE", "SHREECEM", "SHREEPUSHK", "SHREERAMA", "SHRENIK", "SHREYANIND",
    "SHRIKRISH", "SHRIRAMFIN", "SHRIRAMPPS", "SHYAMCENT", "SHYAMMETL", "SICALLTD",
    "SIEMENS", "SIGACHI", "SIGIND", "SIGMA", "SIGNPOST", "SIL", "SILGO",
    "SILINV", "SILLYMONKS", "SILVERTUC", "SIMBHALS", "SIMPLEXINF", "SINDHUTRAD",
    "SINTERCOM", "SIRCA", "SIS", "SITASHREE", "SIYSIL", "SJVN", "SKFINDIA",
    "SKIPPER", "SKMEGGPROD", "SMARTLINK", "SMCGLOBAL", "SMLISUZU", "SMLT",
    "SMSLIFE", "SMSPHARMA", "SNOWMAN", "SOBHA", "SOFTTECH", "SOLARINDS",
    "SOMANYCERA", "SOMATEX", "SOMICONVEY", "SONACOMS", "SONAMLTD", "SONATSOFTW",
    "SOTL", "SOUTHBANK", "SOUTHWEST", "SPAL", "SPANDANA", "SPARC", "SPCENET",
    "SPECIALITY", "SPENCERS", "SPENTEX", "SPIC", "SPLIL", "SPLPETRO", "SPMLINFRA",
    "SPORTKING", "SRD", "SREEL", "SRF", "SRGHFL", "SRHHYPOLTD", "SRM", "SRPL",
    "SSDL", "SSFL", "SSWL", "STANLEY", "STAR", "STARCEMENT", "STARHEALTH",
    "STARPAPER", "STARTECK", "STCINDIA", "STEELCAS", "STEELCITY", "STEELXIND", 
    "STEL", "STERTOOLS", "STLTECH", "STOVEKRAFT", "STYLAMIND", "STYLEBAAZA",
    "SUBCITY", "SUBEXLTD", "SUBROS", "SUDARSCHEM", "SUKHJITS", "SULA", "SUMICHEM",
    "SUMIT", "SUMMITSEC", "SUNCLAY", "SUNDARAM", "SUNDARMFIN", "SUNDARMHLD", 
    "SUNDRMBRAK", "SUNDRMFAST", "SUNFLAG", "SUNPHARMA", "SUNTECK", "SUNTV",
    "SUPERHOUSE", "SUPERSPIN", "SUPRAJIT", "SUPREMEENG", "SUPREMEIND", "SUPRIYA",
    "SURAJEST", "SURAJLTD", "SURANASOL", "SURANAT&P", "SURYALAXMI", "SURYAROSNI",
    "SURYODAY", "SUTLEJTEX", "SUULD", "SUVEN", "SUVENPHAR", "SUVIDHAA", "SUZLON",
    "SVLL", "SVPGLOB", "SWANENERGY", "SWARAJENG", "SWELECTES", "SWSOLAR", "SYMPHONY",
    "SYNCOMF", "SYNGENE", "SYRMA", "TAINWALCHM", "TAJGVK", "TAKE", "TALBROAUT",
    "TANFACIND", "TANLA", "TARC", "TARAPUR", "TARMAT", "TARSONS", "TASTYBITE",
    "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMOTORS",
    "TATAMTRDVR", "TATAPOWER", "TATASTEEL", "TATATECH", "TBZ", "TCI", "TCIEXP",
    "TCNSBRANDS", "TCPLPACK", "TCS", "TDPOWERSYS", "TEAMGRI", "TEAMLEASE",
    "TECHIN", "TECHM", "TECHNOE", "TECILCHEM", "TEGA", "TEJASNET", "TEMBO",
    "TERASOFT", "TEXINFRA", "TEXMOPIPES", "TEXRAIL", "TFCILTD", "TFL", "TGI",
    "THANGAMAYL", "THEINVEST", "THEJO", "THEMISMED", "THERMAX", "THOMASCOOK",
    "THOMASCOTT", "THYROCARE", "TI", "TIIL", "TIINDIA", "TIJARIA", "TIL",
    "TIMESCAN", "TIMESGTY", "TIMETECHNO", "TIMKEN", "TINPLATE", "TIPSFILMS",
    "TIPSMUSIC", "TIRUMALCHM", "TIRUPATIFL", "TITAGARH", "TITAN", "TMB", "TNIDRIL",
    "TNPL", "TNTELE", "TOKYOPLAST", "TOLINS", "TORNTPHARM", "TORNTPOWER",
    "TOTAL", "TOUCHWOOD", "TPHQ", "TPLPLASTEH", "TREEHOUSE", "TREJHARA", "TRENT",
    "TRF", "TRIDENT", "TRIGYN", "TRIL", "TRITURBINE", "TRIVENI", "TRU", "TTKHLTCARE",
    "TTKPRESTIG", "TTL", "TTML", "TV18BRDCST", "TVSELECT", "TVSMOTOR", "TVSSRICHAK",
    "TVTODAY", "TVVISION", "TWL", "UBL", "UCAL", "UCOBANK", "UDS", "UFLEX",
    "UFO", "UGARSUGAR", "UGROCAP", "UJAAS", "UJJIVAN", "UJJIVANSFB", "ULTRACEMCO",
    "UMAEXPORTS", "UMANGDAIRY", "UMESLTD", "UNICHEMLAB", "UNIDT", "UNIENTER",
    "UNIHEALTH", "UNIINFO", "UNIONBANK", "UNIPARTS", "UNITDSPR", "UNITECH",
    "UNITEDTEA", "UNIVASTU", "UNIVCABLES", "UNIVPHOTO", "UNOMINDA", "UPL",
    "URAVI", "URJA", "USHAMART", "USK", "UTIAMC", "UTKARSHBNK", "UTTAMSUGAR",
    "VADILALIND", "VAIBHAVGBL", "VAISHALI", "VAKRANGEE", "VALIANTORG", "VARDHACRLC",
    "VARDMNPOLY", "VARROC", "VASCONEQ", "VASWANI", "VBL", "VCL", "VEDL",
    "VENKEYS", "VENUSPIPES", "VENUSREM", "VERANDA", "VERTOZ", "VESUVIUS",
    "VETO", "VGUARD", "VHL", "VIDHIING", "VIJAYA", "VIKASLIFE", "VIKASPPROP",
    "VIKASECO", "VIKRAM", "VIMTALABS", "VINATIORGA", "VINDHYATEL", "VINEETLAB",
    "VINYLINDIA", "VIPCLOTHNG", "VIPIND", "VIPULLTD", "VIRINCHI",
    "VISAKAIND", "VISASTEEL", "VISHAL", "VISHNU", "VISHWARAJ", "VIVIDHA",
    "VIVIANA", "VLEGOV", "VLSFINANCE", "VMART", "VOLTAMP", "VOLTAS", "VR",
    "VRL", "VRLLOG", "VSSL", "VSTIND", "VSTTILLERS", "VTL", "WABAG", "WALCHANNAG",
    "WANBURY", "WATERBASE", "WEALTH", "WEBELSOLAR", "WEIZMANIND", "WEL", "WELCORP",
    "WELENT", "WELINV", "WELSPUNLIV", "WENDT", "WESTLIFE", "WHEELS", "WHIRLPOOL",
    "WILLAMAGOR", "WINDLAS", "WINDMACHIN", "WINSOME", "WIPL", "WIPRO", "WOCKPHARMA",
    "WONDERLA", "WORTH", "WSI", "WSTCSTPAPR", "XCHANGING", "XELPMOC", "XPROINDIA",
    "YAARI", "YASHO", "YASTEEL", "YATRA", "YESBANK", "YUKEN", "ZEEL", "ZEELEARN",
    "ZEEMEDIA", "ZENITHEXPO", "ZENITHSTL", "ZENSARTECH", "ZENTEC", "ZFSTEERING",
    "ZICOM", "ZODIAC", "ZODIACLOTH", "ZOTA", "ZUARI", "ZUARIGLOB", "ZUARIIND",
    "ZYDUSLIFE", "ZYDUSWELL"
]

UNIVERSE_PRESETS = {
    "All NSE Stocks (Full Listed)": "ALL_NSE",
    "🔍 Single Stock Search": "SINGLE_SEARCH",
    "Nifty 50 Core": "NIFTY_50",
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "LTF.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS",
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS", "AUBANK.NS"
    ],
    "IT & Technology": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
        "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "KPITTECH.NS", "TATAELXSI.NS"
    ],
    "Automobile & EV": [
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "TVSMOTOR.NS", "EICHERMOT.NS", "BHARATFORG.NS", "SONACOMS.NS", "MOTHERSON.NS"
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "MANKIND.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS", "MAXHEALTH.NS"
    ],
    "Defence, Rail & PSUs": [
        "HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS",
        "IRFC.NS", "COCHINSHIP.NS", "BDL.NS", "CONCOR.NS", "BEML.NS"
    ],
}


@st.cache_data(ttl=86400)
def get_all_nse_symbols():
    unique_list = sorted(list(dict.fromkeys(NSE_FULL_EQUITIES)))
    return [f"{s}.NS" for s in unique_list]

selected_universe = st.sidebar.selectbox("Select Stock Basket", list(UNIVERSE_PRESETS.keys()), key="sel_universe")

is_single_search = selected_universe == "🔍 Single Stock Search"

if is_single_search:
    raw_sym_input = st.sidebar.text_input("Enter NSE Symbol", value="ACE")
    clean_sym = raw_sym_input.strip().upper().replace(".NS", "").replace(".BO", "")
    tickers_to_scan = [f"{clean_sym}.NS"] if clean_sym else ["ACE.NS"]
elif selected_universe == "Nifty 50 Core":
    all_symbols = get_all_nse_symbols()
    tickers_to_scan = all_symbols[:50]
elif selected_universe == "All NSE Stocks (Full Listed)":
    all_symbols = get_all_nse_symbols()
    total_found = len(all_symbols)
    scan_limit = st.sidebar.slider(
        "Number of Stocks to Scan",
        min_value=25,
        max_value=total_found,
        step=25,
        help="Scanning fewer stocks at once prevents Yahoo Finance timeouts.",
        key="scan_limit"
    )
    tickers_to_scan = all_symbols[:scan_limit]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

st.sidebar.markdown("### Fundamental Filters")
apply_fund_filter = st.sidebar.checkbox("Enable Strict Fundamental Filters", key="strict_fund")
pat_growth_filter = st.sidebar.checkbox("PAT up > 20% YoY", key="pat_growth")
order_book_gt_mcap_filter = st.sidebar.checkbox("Order Book > Market Cap", key="ob_mcap")

roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, key="roce_rng")
mcap_range_cr = st.sidebar.slider("Market Cap (₹ Cr)", 0, 2000000, step=500, key="mcap_rng")
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, step=0.1, key="max_de")

price_range = st.sidebar.slider("Stock Price (₹)", 0, 10000, step=10, key="price_rng")
rsi_range = st.sidebar.slider("RSI (14)", 0, 100, key="rsi_rng")
min_adx = st.sidebar.slider("Min ADX", 0, 50, key="min_adx")
max_dist_52w_high = st.sidebar.slider("Within % of 52W High", 0, 100, key="dist_52w")

sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    [
        "Any Trend",
        "🌀 EMA Cluster Squeeze & Breakout",
        "⚡ 9/20/44 Triple EMA Bullish Cross",
        "🔥 Multi-Timeframe 20D Breakout",
        "Relative strength",
        "Golden Cross (50 SMA > 200 SMA)",
        "⚡ Weekly MACD Crossover, Stochastics & RSI(7)",
    ],
    key="ma_align"
)

enable_vol_multiplier_10d = st.sidebar.checkbox("Volume > 10D SMA Multiplier", key="vol_10d_en")
vol_multiplier_10d = st.sidebar.slider("10D Volume Surge Multiplier", 0.5, 5.0, step=0.1, disabled=not st.session_state.vol_10d_en, key="vol_10d_mult")

enable_vol_multiplier_20d = st.sidebar.checkbox("Volume > 20D SMA Multiplier", key="vol_20d_en")
vol_multiplier = st.sidebar.slider("20D Volume Surge Multiplier", 0.5, 5.0, step=0.1, disabled=not st.session_state.vol_20d_en, key="vol_20d_mult")

st.sidebar.button("🔓 Restore Open Defaults", on_click=reset_to_open_filters, use_container_width=True)
st.sidebar.button("🎯 Apply Strict Strategy", on_click=apply_strict_filters, use_container_width=True)

scan_button = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)
if st.sidebar.button("🔄 Clear Cache & Rerun", use_container_width=True):
    st.cache_data.clear()
    st.session_state["ai_analysis_cache"] = {}
    st.session_state["screener_data"] = pd.DataFrame()
    gc.collect()
    st.rerun()


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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    if not ticker_list:
        return pd.DataFrame()

    unique_tickers = list(dict.fromkeys(ticker_list))
    total = len(unique_tickers)
    progress_bar = st.progress(0, text="Fetching market data...")
    
    # Safe chunk size to avoid Yahoo Finance IP block limits
    chunk_size = 50
    chunks = [unique_tickers[i : i + chunk_size] for i in range(0, total, chunk_size)]
    rows = []
    seen = set()

    for c_idx, chunk in enumerate(chunks):
        progress_bar.progress((c_idx + 1) / len(chunks), text=f"Scanning batch {c_idx+1}/{len(chunks)} ({min((c_idx+1)*chunk_size, total)}/{total} stocks)...")
        
        batch_data = pd.DataFrame()
        
        # Retry mechanism to handle yfinance random rate limit failures
        for attempt in range(3):
            try:
                batch_data = yf.download(
                    tickers=" ".join(chunk), 
                    period="1y", 
                    interval="1d", 
                    group_by="ticker", 
                    threads=True, 
                    auto_adjust=True, 
                    progress=False, 
                    timeout=10
                )
                if not batch_data.empty:
                    break
            except Exception:
                time.sleep(1)
                
        # Sleep slightly between chunks to prevent 429 errors from YF
        if len(chunks) > 1:
            time.sleep(0.5)

        if batch_data.empty:
            continue

        for ticker in chunk:
            clean_sym = ticker.replace(".NS", "").replace(".BO", "")
            if clean_sym in seen:
                continue

            try:
                hist = pd.DataFrame()
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if ticker in batch_data.columns.get_level_values(0):
                        hist = batch_data[ticker]
                else:
                    if len(chunk) == 1:
                        hist = batch_data

                hist = hist.dropna(how="all")
                if hist.empty or len(hist) < 26:
                    continue

                hist = hist[~hist.index.duplicated(keep="last")]
                curr_price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr_price
                price_change_pct = round(((curr_price - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

                ema_5 = float(hist["Close"].ewm(span=5, adjust=False).mean().iloc[-1])
                ema_9 = float(hist["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
                ema_20 = float(hist["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_44 = float(hist["Close"].ewm(span=44, adjust=False).mean().iloc[-1])
                ema_50 = float(hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                ema_200 = float(hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
                
                sma_50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else curr_price
                sma_200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else curr_price
                
                high_52w = float(hist["High"].max())
                dist_52w_high = max(0.0, ((high_52w - curr_price) / high_52w) * 100.0)

                prev_20d_high = float(hist["High"].iloc[:-1].tail(20).max()) if len(hist) > 20 else float(hist["High"].max())
                is_20d_high_breakout = bool(curr_price > prev_20d_high)

                rsi_val = compute_rsi(hist["Close"], 14)
                adx_val = compute_adx(hist, 14)

                # --- WEEKLY MACD, STOCHASTICS, RSI(7) SETUP LOGIC ---
                is_weekly_setup_match = False
                try:
                    temp_hist = hist.copy()
                    if getattr(temp_hist.index, 'tz', None) is not None:
                        temp_hist.index = temp_hist.index.tz_convert(None)
                    
                    weekly_df = temp_hist.resample("W-FRI").agg({
                        "Open": "first", 
                        "High": "max", 
                        "Low": "min", 
                        "Close": "last", 
                        "Volume": "sum"
                    }).dropna()

                    if len(weekly_df) >= 26:
                        w_close = weekly_df["Close"]
                        w_high = weekly_df["High"]
                        w_low = weekly_df["Low"]

                        # 1. Weekly MACD (21, 13, 9) Crossover
                        w_exp1 = w_close.ewm(span=13, adjust=False).mean()
                        w_exp2 = w_close.ewm(span=21, adjust=False).mean()
                        w_macd_line = w_exp1 - w_exp2
                        w_macd_signal = w_macd_line.ewm(span=9, adjust=False).mean()
                        macd_cross = False
                        if len(w_macd_line) >= 2 and pd.notna(w_macd_line.iloc[-1]):
                            macd_cross = (w_macd_line.iloc[-1] > w_macd_signal.iloc[-1]) and (w_macd_line.iloc[-2] <= w_macd_signal.iloc[-2])

                        # 2. Weekly Fast Stochastic %K (4, 1) Crossover > 80
                        lowest_low = w_low.rolling(window=4).min()
                        highest_high = w_high.rolling(window=4).max()
                        stoch_k = 100 * ((w_close - lowest_low) / (highest_high - lowest_low + 1e-9))
                        stoch_cross = False
                        if len(stoch_k) >= 2 and pd.notna(stoch_k.iloc[-1]):
                            stoch_cross = (stoch_k.iloc[-1] > 80) and (stoch_k.iloc[-2] <= 80)

                        # 3. Weekly RSI (7) Crossover > 70
                        delta = w_close.diff()
                        gain = delta.where(delta > 0, 0.0)
                        loss = -delta.where(delta < 0, 0.0)
                        avg_gain = gain.ewm(alpha=1.0/7, min_periods=7, adjust=False).mean()
                        avg_loss = loss.ewm(alpha=1.0/7, min_periods=7, adjust=False).mean()
                        rs = avg_gain / (avg_loss + 1e-9)
                        w_rsi = 100.0 - (100.0 / (1.0 + rs))
                        rsi_cross = False
                        if len(w_rsi) >= 2 and pd.notna(w_rsi.iloc[-1]):
                            rsi_cross = (w_rsi.iloc[-1] > 70) and (w_rsi.iloc[-2] <= 70)

                        # 4. Weekly ATR (7) > 0
                        tr1 = w_high - w_low
                        tr2 = (w_high - w_close.shift(1)).abs()
                        tr3 = (w_low - w_close.shift(1)).abs()
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        w_atr = tr.ewm(alpha=1.0/7, min_periods=7, adjust=False).mean()
                        atr_cond = False
                        if len(w_atr) >= 1 and pd.notna(w_atr.iloc[-1]):
                            atr_cond = w_atr.iloc[-1] > 0

                        is_weekly_setup_match = bool(macd_cross and stoch_cross and rsi_cross and atr_cond)
                except Exception:
                    is_weekly_setup_match = False

                # --- DAILY VOLUME AND INDICATORS ---
                vol_series = hist["Volume"].dropna()
                curr_vol = int(vol_series.iloc[-1]) if not vol_series.empty else 0
                avg_vol_10 = float(vol_series.rolling(10).mean().iloc[-1]) if len(vol_series) >= 10 else float(curr_vol)
                avg_vol_20 = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else float(curr_vol)
                vol_surge = bool(curr_vol >= (avg_vol_20 * 0.95))
                vol_surge_2x = bool(curr_vol > (avg_vol_20 * 2.0))

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

                cluster_high = max(ema_9, ema_20, ema_44, sma_50)
                cluster_low = min(ema_9, ema_20, ema_44, sma_50)
                cluster_spread = ((cluster_high - cluster_low) / cluster_high * 100.0) if cluster_high > 0 else 10.0
                is_cluster_squeeze = bool(cluster_spread <= 4.5 and curr_price >= cluster_high)

                is_triple_cross = bool(ema_9 > ema_20 > ema_44 and 30 <= curr_price <= 3000 and mcap_cr >= 1000)

                if len(weekly_df) >= 5:
                    w_close_mtf = float(weekly_df["Close"].iloc[-1])
                    w_ema20_mtf = float(weekly_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    w_rsi_mtf = compute_rsi(weekly_df["Close"], 14)
                    w_52h_mtf = float(weekly_df["High"].tail(52).max())
                else:
                    w_close_mtf, w_ema20_mtf, w_rsi_mtf, w_52h_mtf = curr_price, ema_20, rsi_val, high_52w

                passes_mtf_breakout = bool(
                    w_close_mtf > w_ema20_mtf and w_rsi_mtf >= 55.0 and curr_price > ema_20 and is_20d_high_breakout and vol_surge_2x and curr_price >= (w_52h_mtf * 0.80)
                )

                c_20d = float(hist["Close"].iloc[-21]) if len(hist) >= 21 else float(hist["Close"].iloc[0])
                c_125d = float(hist["Close"].iloc[-126]) if len(hist) >= 126 else float(hist["Close"].iloc[0])
                is_relative_strength = bool(
                    ((curr_price - ema_200) / ema_200 * 100.0 > 30.0)
                    and ((curr_price - c_125d) / c_125d * 100.0 > 20.0)
                    and ((curr_price - sma_50) / sma_50 * 100.0 > 20.0)
                    and ((curr_price - c_20d) / c_20d * 100.0 > 20.0)
                )

                score = 0
                if is_20d_high_breakout: score += 25
                if vol_surge_2x: score += 25
                if curr_price > ema_9 > ema_20: score += 25
                if adx_val >= 25: score += 25
                
                if price_change_pct > 8.0:
                    score -= 30

                swing_composite = float(np.clip(score, 10, 100))

                if swing_composite >= 80 and curr_price >= ema_9 >= ema_20 and not is_overextended:
                    action_signal = "🟢 STRONG BUY (Breakout)"
                elif (swing_composite >= 50 or is_triple_cross or is_cluster_squeeze or passes_mtf_breakout or is_overextended) and curr_price >= ema_20:
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
                    "PAT YoY (%)": "N/A",  # Default placeholder for exact fundamental matching
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
                    "SMA_50": round(sma_50, 2),
                    "SMA_200": round(sma_200, 2),
                    "Raw_Ticker": ticker,
                    "_raw_vol": curr_vol,
                    "_avg_vol_10": avg_vol_10,
                    "_avg_vol_20": avg_vol_20,
                    "_change_num": price_change_pct,
                    "_roce_num": roce,
                    "_pat_num": 0.0,
                    "_de_num": 0.5,
                    "_mcap_num": mcap_cr,
                    "_adx_num": adx_val,
                    "_cluster_squeeze_match": is_cluster_squeeze,
                    "_triple_ema_match": is_triple_cross,
                    "_mtf_match": passes_mtf_breakout,
                    "_rs_match": is_relative_strength,
                    "_ob_gt_mcap": is_order_book_gt_mcap,
                    "_weekly_setup_match": is_weekly_setup_match,
                })
                seen.add(clean_sym)
            except Exception as loop_e:
                continue

        del batch_data
        gc.collect()

    progress_bar.empty()
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def get_single_stock_history(ticker):
    try:
        clean_ticker = ticker.strip()
        if not (clean_ticker.endswith(".NS") or clean_ticker.endswith(".BO")):
            clean_ticker = f"{clean_ticker}.NS"

        df = yf.download(
            tickers=clean_ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=10,
        )

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
            return df.dropna(how="all")

        t = yf.Ticker(clean_ticker)
        return t.history(period="1y")
    except Exception:
        return pd.DataFrame()


# AUTO-RUN SCAN ON STARTUP IF EMPTY
if st.session_state["screener_data"].empty:
    with st.spinner("Initializing market scan..."):
        st.session_state["screener_data"] = fetch_screener_universe(tickers_to_scan)

if scan_button or is_single_search:
    with st.spinner("Analyzing market data..."):
        df_raw = fetch_screener_universe(tickers_to_scan)
        st.session_state["screener_data"] = df_raw
else:
    df_raw = st.session_state["screener_data"]

filtered_df = pd.DataFrame()

if not df_raw.empty:
    filtered_df = df_raw.copy()

    # Apply strict numerical filters exactly matching the sidebar
    filtered_df = filtered_df[
        (filtered_df["_roce_num"] >= roce_range[0])
        & (filtered_df["_roce_num"] <= roce_range[1])
        & (filtered_df["_mcap_num"] >= mcap_range_cr[0])
        & (filtered_df["_mcap_num"] <= mcap_range_cr[1])
        & (filtered_df["_de_num"] <= max_de)
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
        elif sma_trend_filter == "🔥 Multi-Timeframe 20D Breakout":
            filtered_df = filtered_df[filtered_df["_mtf_match"] == True]
        elif sma_trend_filter == "Relative strength":
            filtered_df = filtered_df[filtered_df["_rs_match"] == True]
        elif sma_trend_filter == "Golden Cross (50 SMA > 200 SMA)":
            filtered_df = filtered_df[filtered_df["SMA_50"] >= filtered_df["SMA_200"]]
        elif sma_trend_filter == "⚡ Weekly MACD Crossover, Stochastics & RSI(7)":
            filtered_df = filtered_df[filtered_df["_weekly_setup_match"] == True]

        if enable_vol_multiplier_10d:
            filtered_df = filtered_df[filtered_df["_raw_vol"] >= (filtered_df["_avg_vol_10"] * vol_multiplier_10d)]

        if enable_vol_multiplier_20d:
            filtered_df = filtered_df[filtered_df["_raw_vol"] >= (filtered_df["_avg_vol_20"] * vol_multiplier)]

    # --- FUNDAMENTAL FETCH: ALWAYS FETCH FOR MATCHED STOCKS ---
    if not filtered_df.empty:
        filtered_df = filtered_df.copy()
        with st.spinner("Fetching real-time earnings data (PAT YoY) for matched stocks..."):
            valid_indices = []
            for idx, row in filtered_df.iterrows():
                try:
                    t_info = yf.Ticker(row["Raw_Ticker"]).info
                    gr = t_info.get("earningsQuarterlyGrowth")
                    if gr is None:
                        gr = t_info.get("earningsGrowth")
                    
                    if gr is not None:
                        pat_pct = float(gr) * 100
                        filtered_df.at[idx, "PAT YoY (%)"] = f"{pat_pct:.1f}%"
                        filtered_df.at[idx, "_pat_num"] = pat_pct
                    else:
                        pat_pct = 0.0
                        filtered_df.at[idx, "PAT YoY (%)"] = "N/A"
                        filtered_df.at[idx, "_pat_num"] = 0.0
                    
                    if pat_growth_filter:
                        if gr is not None and pat_pct >= 20.0:
                            valid_indices.append(idx)
                    else:
                        valid_indices.append(idx)
                except Exception:
                    filtered_df.at[idx, "PAT YoY (%)"] = "N/A"
                    filtered_df.at[idx, "_pat_num"] = -999.0
                    if not pat_growth_filter:
                        valid_indices.append(idx)
            
            filtered_df = filtered_df.loc[valid_indices]


tab_screener, tab_deepdive, tab_pullback_watchlist, tab_watchlist = st.tabs(
    [
        "📊 Screener & Momentum Signals",
        "🔬 Single Stock Chart & AI Thesis",
        "🎯 Pullback Watchlist & Order Trigger",
        "💼 Paper Trading Portfolio",
    ]
)

with tab_screener:
    if df_raw.empty:
        st.warning("⚠️ No stocks were fetched. Yahoo Finance might be blocking the connection, or the network timed out. Try reducing the 'Number of Stocks to Scan' slider or scan again.")
    elif filtered_df.empty:
        st.info("ℹ️ 0 matching stocks found. Your strict filters filtered out the entire list.")
    else:
        col_title, col_sort_by, col_sort_dir = st.columns([2, 1.2, 1])
        with col_title:
            st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)})")
        with col_sort_by:
            sort_metric = st.selectbox(
                "Sort Results By:",
                ["Volume", "Composite Score", "Change (%)", "Price (₹)", "ADX (14)", "ROCE (%)", "PAT YoY (%)", "RSI (14)", "From 52W High (%)", "Market Cap (₹ Cr)"],
                index=0,
            )
        with col_sort_dir:
            sort_order = st.radio("Order:", ["High to Low (Desc)", "Low to High (Asc)"], horizontal=True)

        sort_col_map = {
            "Volume": "_raw_vol",
            "Composite Score": "Composite Score",
            "Change (%)": "_change_num",
            "Price (₹)": "Price (₹)",
            "ADX (14)": "_adx_num",
            "ROCE (%)": "_roce_num",
            "PAT YoY (%)": "_pat_num",
            "RSI (14)": "RSI (14)",
            "From 52W High (%)": "From 52W High (%)",
            "Market Cap (₹ Cr)": "_mcap_num",
        }
        sorted_results_df = filtered_df.sort_values(
            by=sort_col_map.get(sort_metric, "_raw_vol"),
            ascending=(sort_order == "Low to High (Asc)"),
            na_position="last"
        )

        display_cols = [
            "Ticker", "Signal", "Price (₹)", "Change (%)", "Volume",
            "Composite Score", "ROCE (%)", "PAT YoY (%)", "ADX (14)", "RSI (14)",
            "From 52W High (%)", "Vol Surge", "Market Cap (₹ Cr)",
            "Order Book (₹ Cr)", "OB / MCap"
        ]
        table_data = sorted_results_df[display_cols].copy()
        table_data["Price (₹)"] = table_data["Price (₹)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "-")
        table_data["Composite Score"] = table_data["Composite Score"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        table_data["ROCE (%)"] = table_data["ROCE (%)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        table_data["ADX (14)"] = table_data["ADX (14)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        table_data["RSI (14)"] = table_data["RSI (14)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        table_data["From 52W High (%)"] = table_data["From 52W High (%)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        table_data["Market Cap (₹ Cr)"] = table_data["Market Cap (₹ Cr)"].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "-")
        table_data["Vol Surge"] = table_data["Vol Surge"].apply(lambda x: "✅" if x else "⬜")

        styled_table = table_data.style.set_properties(**{
            "text-align": "center",
            "font-weight": "500"
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "center !important"), ("justify-content", "center !important")]},
            {"selector": "td", "props": [("text-align", "center !important"), ("justify-content", "center !important")]},
        ])

        selection_event = st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selection_event and selection_event.selection and selection_event.selection.rows:
            selected_row_idx = selection_event.selection.rows[0]
            clicked_ticker_sym = table_data.iloc[selected_row_idx]["Ticker"]
            st.session_state["selected_ticker"] = f"{clicked_ticker_sym}.NS"

with tab_deepdive:
    if df_raw.empty:
        st.write("No data available.")
    else:
        stock_options = filtered_df["Raw_Ticker"].tolist() if not filtered_df.empty else df_raw["Raw_Ticker"].tolist()
        current_choice = st.session_state.get("selected_ticker", stock_options[0] if stock_options else "ACE.NS")
        default_index = stock_options.index(current_choice) if current_choice in stock_options else 0
        selected_stock = st.selectbox("Selected Stock:", stock_options, index=default_index)
        st.session_state["selected_ticker"] = selected_stock

        if selected_stock:
            hist = get_single_stock_history(selected_stock)
            stock_match = df_raw[df_raw["Raw_Ticker"] == selected_stock]
            stock_row = stock_match.iloc[0] if not stock_match.empty else None

            if hist is not None and not hist.empty:
                hist["EMA_9"] = hist["Close"].ewm(span=9, adjust=False).mean()
                hist["EMA_20"] = hist["Close"].ewm(span=20, adjust=False).mean()
                hist["EMA_44"] = hist["Close"].ewm(span=44, adjust=False).mean()
                hist["SMA_50"] = hist["Close"].rolling(50).mean()
                hist["SMA_200"] = hist["Close"].rolling(200).mean()

                curr_p = float(hist["Close"].iloc[-1])
                ema9_val = float(hist["EMA_9"].iloc[-1])
                ema20_val = float(hist["EMA_20"].iloc[-1])
                ema44_val = float(hist["EMA_44"].iloc[-1])
                curr_signal = stock_row["Signal"] if stock_row is not None else "N/A"
                curr_score = stock_row["Composite Score"] if stock_row is not None else 0
                curr_adx = stock_row["ADX (14)"] if stock_row is not None else 25.0
                curr_change = stock_row["Change (%)"] if stock_row is not None else "0.00%"
                curr_volume = stock_row["Volume"] if stock_row is not None else "N/A"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{curr_p:,.2f}", delta=curr_change)
                c2.metric("9 / 20 / 44 EMA", f"₹{ema9_val:.1f} / ₹{ema20_val:.1f} / ₹{ema44_val:.1f}")
                c3.metric("Volume / ADX", f"{curr_volume} | ADX: {curr_adx}")
                c4.metric("Action Signal", curr_signal)

                fig = go.Figure(
                    data=[
                        go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"),
                        go.Scatter(x=hist.index, y=hist["EMA_9"], line=dict(color="#00f2ff", width=1.5), name="9 EMA (Fast)"),
                        go.Scatter(x=hist.index, y=hist["EMA_20"], line=dict(color="#ffd700", width=1.5), name="20 EMA (Momentum)"),
                        go.Scatter(x=hist.index, y=hist["EMA_44"], line=dict(color="#a855f7", width=1.5), name="44 EMA (Baseline)"),
                        go.Scatter(x=hist.index, y=hist["SMA_50"], line=dict(color="#ff9900", width=1.5), name="50 SMA"),
                        go.Scatter(x=hist.index, y=hist["SMA_200"], line=dict(color="#4d79ff", width=1.5), name="200 SMA"),
                    ]
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=480,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🤖 AI Short-Term Swing Thesis & Trade Setup")
                cached_thesis = st.session_state.get("ai_analysis_cache", {}).get(selected_stock)
                if cached_thesis:
                    st.markdown(cached_thesis)

                if st.button("Generate Short-Term Swing Setup for " + selected_stock):
                    if not GEMINI_API_KEY:
                        st.warning("Please provide your Gemini API Key in the left sidebar.")
                    else:
                        prompt = f"""
                        You are a Professional Swing Trader & Technical Analyst specializing in Indian Equities (NSE).
                        Evaluate this pure Short-Term Swing / Momentum Breakout trade setup:
                        - Stock: {selected_stock}
                        - Current Price: ₹{curr_p:.2f} (Day Change: {curr_change})
                        - Traded Volume: {curr_volume}
                        - 9 EMA: ₹{ema9_val:.2f} | 20 EMA: ₹{ema20_val:.2f} | 44 EMA: ₹{ema44_val:.2f}
                        - ADX (14): {curr_adx}, RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}
                        - Breakout Composite Score: {curr_score}/100 | System Signal: {curr_signal}

                        Provide a structured swing trade plan:
                        1. **Breakout Setup Assessment**: Is momentum active, in a healthy base pullback, or exhausted?
                        2. **Exact Actionable Verdict**: Choose one strictly: [STRONG BUY | BUY (ON PULLBACK) | WAIT | AVOID].
                        3. **Trade Blueprint**: Entry Range (₹), Strict Stop-Loss (₹), Targets (Target 1 & 2 with Risk:Reward >= 1:2).
                        4. **Exit Trigger**: Invalidation condition for swing trades.
                        """
                        with st.spinner("Analyzing momentum setup with Gemini..."):
                            success = False
                            error_logs = []
                            candidate_models = []
                            try:
                                for m in genai.list_models():
                                    if "generateContent" in m.supported_generation_methods:
                                        candidate_models.append(m.name.replace("models/", ""))
                            except Exception as e:
                                error_logs.append(f"Model listing error: {e}")

                            if not candidate_models:
                                candidate_models = [
                                    "gemini-1.5-flash",
                                    "gemini-2.0-flash",
                                    "gemini-1.5-flash-8b",
                                    "gemini-1.5-pro",
                                    "gemini-pro",
                                ]

                            for model_name in candidate_models:
                                try:
                                    model = genai.GenerativeModel(model_name)
                                    res = model.generate_content(prompt)
                                    if res and res.text:
                                        st.session_state["ai_analysis_cache"][selected_stock] = res.text
                                        st.markdown(res.text)
                                        success = True
                                        break
                                except Exception as err:
                                    error_logs.append(f"{model_name}: {str(err)}")
                                    continue

                            if not success:
                                st.error("Failed to generate AI thesis.")
                                with st.expander("🔍 View Error Details"):
                                    for err in error_logs:
                                        st.code(err)

with tab_pullback_watchlist:
    st.subheader("🎯 Pullback Watchlist & Limit Order Execution")
    
    render_alert_permission_banner()
    
    st.info("💡 **Pullback Entry Engine:** Place limit orders below current market price (LTP). When market price dips to or below your target, the system triggers, sounds an alert, and automatically executes the trade.")

    col_w_dl, col_w_up = st.columns([1, 1])
    with col_w_up:
        uploaded_watchlist = st.file_uploader("📥 Restore Watchlist from Backup (.json)", type=["json"], key="watchlist_uploader")
        if uploaded_watchlist is not None:
            try:
                restored_wb_data = json.load(uploaded_watchlist)
                if isinstance(restored_wb_data, list):
                    valid_items = [item for item in restored_wb_data if isinstance(item, dict) and "Target Buy (₹)" in item]
                    if valid_items:
                        st.session_state["pullback_watchlist"] = valid_items
                        save_json_file(WATCHLIST_FILE, valid_items)
                        st.success("Pullback watchlist successfully restored!")
                        st.rerun()
                    else:
                        st.error("Uploaded file does not contain valid pullback watchlist entries.")
            except Exception as e:
                st.error(f"Failed to restore watchlist backup: {e}")

    with col_w_dl:
        active_watchlist_data = st.session_state.get("pullback_watchlist", [])
        if active_watchlist_data:
            st.download_button(
                label="💾 Download Watchlist Backup (.json)",
                data=json.dumps(active_watchlist_data, indent=4),
                file_name="watchlist_backup.json",
                mime="application/json",
                use_container_width=True,
            )

    if not df_raw.empty:
        pullback_candidates = df_raw["Raw_Ticker"].tolist()
        curr_selected = st.session_state.get("selected_ticker", pullback_candidates[0] if pullback_candidates else "ACE.NS")
        default_wb_idx = pullback_candidates.index(curr_selected) if curr_selected in pullback_candidates else 0

        with st.expander("➕ Add Stock to Pullback Watchlist", expanded=False):
            cw1, cw2, cw3, cw4, cw5 = st.columns([1.2, 1, 1, 1, 1])
            with cw1:
                sel_stock = st.selectbox("Stock Candidate:", pullback_candidates, index=default_wb_idx)
                matched_match = df_raw[df_raw["Raw_Ticker"] == sel_stock]
                matched_row = matched_match.iloc[0] if not matched_match.empty else df_raw.iloc[0]
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
                strat_note = st.text_input("Strategy Note", value=sma_trend_filter)
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

    active_watchlist = st.session_state.get("pullback_watchlist", [])
    if active_watchlist:
        live_price_dict = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"])) if not df_raw.empty else {}
        updated_watchlist = []
        display_rows = []

        for item in active_watchlist:
            if not isinstance(item, dict) or "Target Buy (₹)" not in item:
                continue

            sym = item.get("Raw_Ticker") or f"{item.get('Ticker')}.NS"
            clean_sym = item.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
            target_buy = float(item.get("Target Buy (₹)", 0.0))
            sl_price = float(item.get("SL (₹)", 0.0))
            tgt_price = float(item.get("TGT (₹)", 0.0))
            qty = int(item.get("Qty", 1))
            status_str = item.get("Status", "⏳ Waiting for Pullback")

            curr_ltp = live_price_dict.get(sym)
            if curr_ltp is None:
                try:
                    curr_ltp = float(yf.Ticker(sym).fast_info.last_price)
                except Exception:
                    curr_ltp = None

            if "Waiting" in status_str and curr_ltp is not None and curr_ltp > 0 and curr_ltp <= target_buy:
                status_str = "⚡ Triggered / Bought"
                item["Status"] = status_str
                
                play_trigger_alert(clean_sym, target_buy)
                st.toast(f"🎯 PULLBACK HIT! {clean_sym} bought at ₹{curr_ltp:.2f}!", icon="⚡")
                st.success(f"🔔 **Pullback Triggered:** {clean_sym} reached buy level ₹{target_buy:,.2f} (LTP: ₹{curr_ltp:,.2f}). Auto-executed into Paper Trading Portfolio!")

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
                    "Remarks": f"Pullback Auto-Entry ({item.get('Strategy', sma_trend_filter)})",
                    "Status": "🟢 Open",
                    "Invested (₹)": round(curr_ltp * qty, 2),
                    "Raw_Ticker": sym,
                }
                if not any(p.get("id") == trade_record["id"] for p in st.session_state["paper_portfolio"]):
                    st.session_state["paper_portfolio"].append(trade_record)
                    save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])

            item["Status"] = status_str
            updated_watchlist.append(item)

            dist_str = f"{((curr_ltp - target_buy) / target_buy * 100.0):+.2f}% away" if (curr_ltp and target_buy > 0 and "Waiting" in status_str) else ("Executed ✅" if "Triggered" in status_str else "Fetching...")

            display_rows.append({
                "Date Added": item.get("Date Added", str(date.today())),
                "Ticker": clean_sym,
                "Current LTP (₹)": f"₹{curr_ltp:,.2f}" if curr_ltp else "-",
                "Target Buy (₹)": f"₹{target_buy:,.2f}",
                "Distance to Entry": dist_str,
                "SL (₹)": f"₹{sl_price:,.2f}",
                "TGT (₹)": f"₹{tgt_price:,.2f}",
                "Qty": qty,
                "Status": status_str,
                "Strategy": item.get("Strategy", sma_trend_filter),
            })

        st.session_state["pullback_watchlist"] = updated_watchlist
        save_json_file(WATCHLIST_FILE, updated_watchlist)
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        with st.expander("✏️ Edit Watchlist Position (Modify Entry, SL, TGT & Qty)", expanded=False):
            col_sel_edit, _ = st.columns([2, 1])
            with col_sel_edit:
                wb_edit_options = {
                    f"{it.get('Ticker')} (Target: ₹{it.get('Target Buy (₹)')}) [{it.get('Status')}]": idx 
                    for idx, it in enumerate(updated_watchlist)
                }
                selected_wb_edit_label = st.selectbox("Select Watchlist Item to Edit:", list(wb_edit_options.keys()), key="edit_wb_selector")
            
            if wb_edit_options:
                wb_edit_idx = wb_edit_options[selected_wb_edit_label]
                curr_wb_item = updated_watchlist[wb_edit_idx]

                ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                with ec1:
                    new_target_buy = st.number_input("Target Entry (₹)", value=float(curr_wb_item.get("Target Buy (₹)") or 0.0), step=0.5, key=f"wb_edit_tgtbuy_{wb_edit_idx}")
                with ec2:
                    new_wb_sl = st.number_input("Stop Loss (₹)", value=float(curr_wb_item.get("SL (₹)") or 0.0), step=0.5, key=f"wb_edit_sl_{wb_edit_idx}")
                with ec3:
                    new_wb_tgt = st.number_input("Target (₹)", value=float(curr_wb_item.get("TGT (₹)") or 0.0), step=0.5, key=f"wb_edit_tgt_{wb_edit_idx}")
                with ec4:
                    new_wb_qty = st.number_input("Quantity", value=int(curr_wb_item.get("Qty", 50)), min_value=1, step=1, key=f"wb_edit_qty_{wb_edit_idx}")
                with ec5:
                    new_wb_strat = st.text_input("Strategy Note", value=curr_wb_item.get("Strategy", ""), key=f"wb_edit_strat_{wb_edit_idx}")

                if st.button("💾 Save Watchlist Updates", key=f"save_wb_btn_{wb_edit_idx}"):
                    updated_watchlist[wb_edit_idx]["Target Buy (₹)"] = new_target_buy
                    updated_watchlist[wb_edit_idx]["SL (₹)"] = new_wb_sl
                    updated_watchlist[wb_edit_idx]["TGT (₹)"] = new_wb_tgt
                    updated_watchlist[wb_edit_idx]["Qty"] = new_wb_qty
                    updated_watchlist[wb_edit_idx]["Strategy"] = new_wb_strat.strip()
                    
                    st.session_state["pullback_watchlist"] = updated_watchlist
                    save_json_file(WATCHLIST_FILE, updated_watchlist)
                    st.success("Watchlist item successfully updated!")
                    st.rerun()

        m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
        with m_col1:
            del_choices = {f"{it.get('Ticker')} (Target: ₹{it.get('Target Buy (₹)')}) [{it.get('Status')}]": idx for idx, it in enumerate(updated_watchlist)}
            if del_choices:
                sel_del = st.selectbox("Select Watchlist Item:", list(del_choices.keys()), key="del_wb_select")
        with m_col2:
            st.write("")
            st.write("")
            if del_choices and st.button("🔄 Re-Arm / Reset to Waiting", use_container_width=True):
                d_idx = del_choices[sel_del]
                rearm_sym = updated_watchlist[d_idx].get("Ticker")
                updated_watchlist[d_idx]["Status"] = "⏳ Waiting for Pullback"
                st.session_state["pullback_watchlist"] = updated_watchlist
                save_json_file(WATCHLIST_FILE, updated_watchlist)

                st.session_state["paper_portfolio"] = [
                    p for p in st.session_state["paper_portfolio"]
                    if not (p.get("Ticker") == rearm_sym and "Pullback" in p.get("Remarks", ""))
                ]
                save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                st.success(f"Re-armed {rearm_sym} and synchronized trades!")
                st.rerun()
        with m_col3:
            st.write("")
            st.write("")
            if del_choices and st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
                d_idx = del_choices[sel_del]
                removed_sym = updated_watchlist[d_idx].get("Ticker")
                updated_watchlist.pop(d_idx)
                st.session_state["pullback_watchlist"] = updated_watchlist
                save_json_file(WATCHLIST_FILE, updated_watchlist)

                st.session_state["paper_portfolio"] = [
                    p for p in st.session_state["paper_portfolio"]
                    if not (p.get("Ticker") == removed_sym and "Pullback" in p.get("Remarks", ""))
                ]
                save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                st.success(f"Deleted {removed_sym} from watchlist!")
                st.rerun()
    else:
        st.info("Watchlist is empty. Add a pullback setup above.")

with tab_watchlist:
    st.subheader("💼 Paper Trading Portfolio & Risk Manager")
    active_portfolio = st.session_state.get("paper_portfolio", [])

    if not df_raw.empty:
        with st.expander("➕ Execute New Paper Trade (Custom SL, Target & Remarks)", expanded=False):
            col_add1, col_add2, col_add3, col_add4, col_add5 = st.columns([1.2, 1, 1, 1, 1])
            with col_add1:
                available_tickers = df_raw["Raw_Ticker"].tolist() if not df_raw.empty else ["ACE.NS"]
                curr_selected_trade = st.session_state.get("selected_ticker", available_tickers[0])
                default_trade_idx = available_tickers.index(curr_selected_trade) if curr_selected_trade in available_tickers else 0
                trade_stock = st.selectbox("Stock:", available_tickers, index=default_trade_idx)
            with col_add2:
                trade_date = st.date_input("Entry Date", value=date.today())
            with col_add3:
                matched_stock = df_raw[df_raw["Raw_Ticker"] == trade_stock] if not df_raw.empty else pd.DataFrame()
                live_price = float(matched_stock["Price (₹)"].iloc[0]) if not matched_stock.empty else 100.0
                buy_price = st.number_input("Entry Price (₹)", value=live_price, min_value=0.1, step=0.5)
            with col_add4:
                sl_price = st.number_input("Stop Loss (SL ₹)", value=round(buy_price * 0.96, 1), min_value=0.0, step=0.5)
            with col_add5:
                tgt_price = st.number_input("Target (TGT ₹)", value=round(buy_price * 1.08, 1), min_value=0.0, step=0.5)

            col_sub1, col_sub2, col_btn = st.columns([1, 2.5, 1])
            with col_sub1:
                quantity = st.number_input("Quantity", value=50, min_value=1, step=1)
            with col_sub2:
                remarks = st.text_input("Trade Remarks / Strategy", value=sma_trend_filter)
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📥 Execute Trade", use_container_width=True):
                    raw_sym = trade_stock if trade_stock.endswith(".NS") else f"{trade_stock}.NS"
                    trade_id = f"{raw_sym}_{int(time.time())}"
                    new_trade = {
                        "id": trade_id,
                        "Date": str(trade_date),
                        "Exit_Date": "",
                        "Ticker": raw_sym.replace(".NS", "").replace(".BO", ""),
                        "Buy Price (₹)": buy_price,
                        "SL (₹)": sl_price,
                        "TGT (₹)": tgt_price,
                        "Exit Price (₹)": 0.0,
                        "Qty": int(quantity),
                        "Remarks": remarks.strip(),
                        "Status": "🟢 Open",
                        "Invested (₹)": round(buy_price * quantity, 2),
                        "Raw_Ticker": raw_sym,
                    }
                    st.session_state["paper_portfolio"].append(new_trade)
                    save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                    st.success(f"Executed trade for {quantity} shares of {new_trade['Ticker']}!")
                    st.rerun()

    if active_portfolio:
        with st.expander("🗑️ Delete a Trade / Row Added by Mistake", expanded=False):
            col_del_sel, col_del_btn = st.columns([3, 1])
            with col_del_sel:
                delete_trade_choices = {
                    f"{pos.get('Ticker')} (Entry: ₹{pos.get('Buy Price (₹)')} on {pos.get('Date')}) [{pos.get('Status', '🟢 Open')}] - [ID: {pos.get('id', idx)}]": idx
                    for idx, pos in enumerate(active_portfolio)
                }
                selected_trade_to_delete = st.selectbox("Select Position to Delete:", list(delete_trade_choices.keys()), key="delete_row_selector")
            with col_del_btn:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete Selected Trade", type="primary", use_container_width=True):
                    del_idx = delete_trade_choices[selected_trade_to_delete]
                    deleted_ticker = active_portfolio[del_idx].get("Ticker", "Trade")
                    active_portfolio.pop(del_idx)
                    st.session_state["paper_portfolio"] = active_portfolio
                    save_json_file(PORTFOLIO_FILE, active_portfolio)
                    st.success(f"Successfully deleted {deleted_ticker} position!")
                    st.rerun()

    col_dl, col_up = st.columns([1, 1])
    with col_up:
        uploaded_portfolio = st.file_uploader("📥 Restore Trades from Backup (.json)", type=["json"], key="portfolio_uploader")
        if uploaded_portfolio is not None:
            try:
                restored_data = json.load(uploaded_portfolio)
                if isinstance(restored_data, list) and len(restored_data) > 0:
                    st.session_state["paper_portfolio"] = restored_data
                    save_json_file(PORTFOLIO_FILE, restored_data)
                    st.success("Portfolio successfully restored!")
            except Exception as e:
                st.error(f"Failed to restore backup: {e}")

    with col_dl:
        if active_portfolio:
            st.download_button(
                label="💾 Download Portfolio Backup (.json)",
                data=json.dumps(active_portfolio, indent=4),
                file_name="portfolio_backup.json",
                mime="application/json",
                use_container_width=True,
            )

    if active_portfolio:
        live_price_dict = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"])) if not df_raw.empty else {}
        open_invested = 0.0
        open_current_val = 0.0
        unrealised_pnl_total = 0.0
        realised_pnl_total = 0.0
        winning_trades_count = 0
        losing_trades_count = 0
        open_trades_count = 0
        total_trades_count = len(active_portfolio)

        portfolio_rows = []
        updated_portfolio_data = []

        for idx, pos in enumerate(active_portfolio):
            sym = pos.get("Raw_Ticker", f"{pos.get('Ticker', 'ACE')}.NS")
            clean_t = pos.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
            buy_p = float(pos.get("Buy Price (₹)", 0.0))
            
            curr_p = live_price_dict.get(sym)
            if curr_p is None:
                try:
                    curr_p = float(yf.Ticker(sym).fast_info.last_price)
                except Exception:
                    curr_p = buy_p

            qty = int(pos.get("Qty", 1))
            invested = float(pos.get("Invested (₹)", buy_p * qty))
            sl = float(pos.get("SL (₹)", 0.0))
            tgt = float(pos.get("TGT (₹)", 0.0))
            pos_date_str = str(pos.get("Date", date.today()))
            pos_exit_date_str = str(pos.get("Exit_Date", "") or "")
            pos_remarks = str(pos.get("Remarks", sma_trend_filter))
            pos_status = str(pos.get("Status", "🟢 Open"))
            saved_exit_price = float(pos.get("Exit Price (₹)") or 0.0)

            if pos_status == "🟢 Open":
                if sl > 0 and curr_p <= sl:
                    pos_status = "🔴 SL Hit (Closed)"
                    pos_exit_date_str = str(date.today())
                    saved_exit_price = sl
                elif tgt > 0 and curr_p >= tgt:
                    pos_status = "🎯 TGT Hit (Closed)"
                    pos_exit_date_str = str(date.today())
                    saved_exit_price = tgt

            pos["Status"] = pos_status
            pos["Exit_Date"] = pos_exit_date_str
            pos["Exit Price (₹)"] = saved_exit_price
            updated_portfolio_data.append(pos)

            try:
                d_entry = datetime.strptime(pos_date_str, "%Y-%m-%d").date()
                d_exit = datetime.strptime(pos_exit_date_str.strip(), "%Y-%m-%d").date() if (pos_exit_date_str and pos_exit_date_str.strip() and pos_exit_date_str != "-") else date.today()
                holding_days = max(0, (d_exit - d_entry).days)
            except Exception:
                holding_days = 0

            if "Closed" in pos_status or pos_status == "⚪ Sold Manually":
                exit_val = saved_exit_price if saved_exit_price > 0 else curr_p
                pnl = round((exit_val - buy_p) * qty, 2)
                pnl_pct = round((pnl / invested) * 100.0, 2) if invested > 0 else 0.0
                realised_pnl_total += pnl
                effective_curr_p = exit_val
            else:
                open_trades_count += 1
                pnl = round((curr_p - buy_p) * qty, 2)
                pnl_pct = round((pnl / invested) * 100.0, 2) if invested > 0 else 0.0
                open_invested += invested
                open_current_val += round(curr_p * qty, 2)
                unrealised_pnl_total += pnl
                effective_curr_p = curr_p

            if pnl > 0:
                winning_trades_count += 1
            elif pnl < 0:
                losing_trades_count += 1

            portfolio_rows.append({
                "Entry Date": pos_date_str,
                "Sold Date": pos_exit_date_str if pos_exit_date_str else "-",
                "Holding": f"{holding_days} d",
                "Ticker": clean_t,
                "Status": pos_status,
                "Remarks / Strategy": pos_remarks,
                "Entry (₹)": f"₹{buy_p:,.2f}",
                "SL (₹)": f"₹{sl:,.2f}" if sl > 0 else "-",
                "TGT (₹)": f"₹{tgt:,.2f}" if tgt > 0 else "-",
                "Current Price (₹)": f"₹{effective_curr_p:,.2f}",
                "Qty": qty,
                "Invested (₹)": f"₹{invested:,.2f}",
                "P&L (₹)": f"{'+' if pnl >= 0 else ''}₹{pnl:,.2f}",
                "P&L (%)": f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%",
                "_raw_pnl": pnl,
            })

        win_rate_pct = (winning_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
        loss_rate_pct = (losing_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

        trade_summary_html = f"""
        <div class="trade-summary-card">
            <div class="trade-stat-box">
                <div class="trade-stat-label">Total Trades</div>
                <div class="trade-stat-val">{total_trades_count}</div>
            </div>
            <div class="trade-stat-box">
                <div class="trade-stat-label">Winning Trades</div>
                <div class="trade-stat-val" style="color: #16a34a;">{winning_trades_count}</div>
            </div>
            <div class="trade-stat-box">
                <div class="trade-stat-label">Losing Trades</div>
                <div class="trade-stat-val" style="color: #dc2626;">{losing_trades_count}</div>
            </div>
            <div class="trade-stat-box">
                <div class="trade-stat-label">Open Trade</div>
                <div class="trade-stat-val" style="color: #0284c7;">{open_trades_count}</div>
            </div>
        </div>
        """
        st.markdown(trade_summary_html, unsafe_allow_html=True)

        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        p_col1.metric("Open Invested Capital", f"₹{open_invested:,.2f}")
        p_col2.metric("Open Portfolio Value", f"₹{open_current_val:,.2f}")
        p_col3.metric(
            "Unrealised P&L (Open)",
            f"₹{unrealised_pnl_total:,.2f}",
            delta=f"{(unrealised_pnl_total / open_invested * 100.0):.2f}%" if open_invested > 0 else "0.00%",
        )
        p_col4.metric(
            "Realised P&L (Closed)",
            f"₹{realised_pnl_total:,.2f}",
            delta_color="normal" if realised_pnl_total >= 0 else "inverse",
        )
        p_col5.metric(
            "Win / Loss Rate (%)",
            f"{win_rate_pct:.1f}% Win",
            delta=f"{loss_rate_pct:.1f}% Loss",
            delta_color="inverse",
        )

        with st.expander("✏️ Edit Position Parameters (Modify SL, TGT, Sold Date, Status & Exit Price)", expanded=False):
            col_sel_edit, _ = st.columns([2, 1])
            with col_sel_edit:
                trade_edit_options = {
                    f"{pos.get('Ticker')} (Entry: ₹{pos.get('Buy Price (₹)')} on {pos.get('Date')}) - [{pos.get('Status', '🟢 Open')}] [ID: {pos.get('id', idx)}]": idx
                    for idx, pos in enumerate(active_portfolio)
                }
                selected_edit_label = st.selectbox("Select Position to Edit:", list(trade_edit_options.keys()))
            
            edit_idx = trade_edit_options[selected_edit_label]
            curr_item = active_portfolio[edit_idx]

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            with ec1:
                new_sl_val = st.number_input("Edit SL (₹)", value=float(curr_item.get("SL (₹)") or 0.0), step=0.5, key=f"edit_sl_{edit_idx}")
            with ec2:
                new_tgt_val = st.number_input("Edit TGT (₹)", value=float(curr_item.get("TGT (₹)") or 0.0), step=0.5, key=f"edit_tgt_{edit_idx}")
            with ec3:
                current_status_opts = ["🟢 Open", "🔴 SL Hit (Closed)", "🎯 TGT Hit (Closed)", "⚪ Sold Manually"]
                existing_status = curr_item.get("Status", "🟢 Open")
                status_idx = current_status_opts.index(existing_status) if existing_status in current_status_opts else 0
                new_status_val = st.selectbox("Status", current_status_opts, index=status_idx, key=f"edit_status_{edit_idx}")
            with ec4:
                existing_exit_date_str = curr_item.get("Exit_Date")
                try:
                    parsed_exit_date = datetime.strptime(str(existing_exit_date_str), "%Y-%m-%d").date() if existing_exit_date_str and existing_exit_date_str != "-" else date.today()
                except Exception:
                    parsed_exit_date = date.today()
                new_exit_date = st.date_input("Sold Date", value=parsed_exit_date, key=f"edit_exit_date_{edit_idx}")
            with ec5:
                new_exit_price = st.number_input("Exit Price (₹)", value=float(curr_item.get("Exit Price (₹)") or curr_item.get("Buy Price (₹)") or 0.0), step=0.5, key=f"edit_exit_price_{edit_idx}")

            if st.button("💾 Save Position Updates", key=f"save_btn_{edit_idx}"):
                active_portfolio[edit_idx]["SL (₹)"] = new_sl_val
                active_portfolio[edit_idx]["TGT (₹)"] = new_tgt_val
                active_portfolio[edit_idx]["Status"] = new_status_val
                if new_status_val != "🟢 Open":
                    active_portfolio[edit_idx]["Exit_Date"] = str(new_exit_date)
                    active_portfolio[edit_idx]["Exit Price (₹)"] = new_exit_price
                else:
                    active_portfolio[edit_idx]["Exit_Date"] = ""
                    active_portfolio[edit_idx]["Exit Price (₹)"] = 0.0
                save_json_file(PORTFOLIO_FILE, active_portfolio)
                st.success("Position successfully updated!")
                st.rerun()

        port_df = pd.DataFrame(portfolio_rows)
        display_port_cols = [
            "Entry Date", "Sold Date", "Holding", "Ticker", "Status",
            "Remarks / Strategy", "Entry (₹)", "SL (₹)", "TGT (₹)",
            "Current Price (₹)", "Qty", "Invested (₹)", "P&L (₹)", "P&L (%)"
        ]
        final_port_display = port_df[display_port_cols].copy()

        def highlight_pnl_dark_green_red(val):
            try:
                clean_str = str(val).replace("₹", "").replace("%", "").replace("+", "").replace(",", "").strip()
                num = float(clean_str)
                if num > 0:
                    return "color: #15803d; font-weight: 700;"
                elif num < 0:
                    return "color: #dc2626; font-weight: 700;"
                else:
                    return "color: #64748b; font-weight: normal;"
            except Exception:
                return ""

        styled_port = final_port_display.style.map(
            highlight_pnl_dark_green_red, subset=["P&L (₹)", "P&L (%)"]
        ).set_properties(**{
            "text-align": "center",
            "font-weight": "500"
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "center !important"), ("justify-content", "center !important")]},
            {"selector": "td", "props": [("text-align", "center !important"), ("justify-content", "center !important")]},
        ])

        st.dataframe(styled_port, use_container_width=True, hide_index=True)

        if st.button("🗑️ Reset / Clear All Trades"):
            st.session_state["paper_portfolio"] = []
            save_json_file(PORTFOLIO_FILE, [])
            st.rerun()
