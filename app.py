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

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Indian Market AI Stock Screener & Paper Trading",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
[data-testid="stDataEditor"] td, [data-testid="stDataEditor"] th {
    text-align: center !important; vertical-align: middle !important;
}
div[data-testid="stDataFrame"] div[role="columnheader"],
div[data-testid="stDataEditor"] div[role="columnheader"],
div[data-testid="stDataFrame"] div[role="gridcell"],
div[data-testid="stDataEditor"] div[role="gridcell"] {
    text-align: center !important; justify-content: center !important;
}
.index-ticker-container {
    display: flex; flex-wrap: wrap; justify-content: center;
    background-color: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 14px; margin-bottom: 18px;
    gap: 12px 16px; align-items: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
@media (prefers-color-scheme: dark) {
    .index-ticker-container { background-color: #1e293b; border-color: #334155; }
}
.index-item { display: flex; align-items: center; gap: 8px; white-space: nowrap; font-size: 13.5px; font-weight: 500; }
.index-name { color: #64748b; font-weight: 600; }
.index-val { font-weight: 700; }
.index-pos { color: #16a34a; font-weight: 600; }
.index-neg { color: #dc2626; font-weight: 600; }
.index-divider { color: #cbd5e1; }
.trade-summary-card {
    display: flex; justify-content: space-around; align-items: center;
    background: #ffffff; border: 2px solid #0284c7; border-radius: 8px;
    padding: 10px 14px; margin: 12px 0 18px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
@media (prefers-color-scheme: dark) {
    .trade-summary-card { background: #0f172a; border-color: #38bdf8; }
}
.trade-stat-box { text-align: center; flex: 1; border-right: 1px solid #e2e8f0; }
@media (prefers-color-scheme: dark) { .trade-stat-box { border-right-color: #334155; } }
.trade-stat-box:last-child { border-right: none; }
.trade-stat-label { font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
@media (prefers-color-scheme: dark) { .trade-stat-label { color: #f8fafc; } }
.trade-stat-val { font-size: 20px; font-weight: 800; color: #0369a1; }
@media (prefers-color-scheme: dark) { .trade-stat-val { color: #38bdf8; } }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ALERT HELPERS
# ============================================================
def render_alert_permission_banner():
    banner_html = """
    <div style="display:flex;align-items:center;justify-content:space-between;background:#ecfdf5;border:2px solid #10b981;border-radius:10px;padding:12px 18px;margin-bottom:14px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;">🔔</span>
            <div>
                <div style="font-size:14.5px;font-weight:700;color:#065f46;">Auto-Market Hours Alert Engine (9:15 AM – 3:30 PM IST)</div>
                <div style="font-size:12px;color:#047857;" id="market-time-status">Checking...</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <button id="alert-btn" onclick="activateSystemAlerts()" style="background:#059669;color:#fff;border:none;border-radius:8px;padding:10px 18px;font-size:13.5px;font-weight:700;cursor:pointer;">
                🔊 Grant Sound & Push Permission
            </button>
            <span id="alert-status-msg" style="font-size:13px;font-weight:700;color:#065f46;"></span>
        </div>
    </div>
    <script>
    function checkMarketHoursAndPermissions() {
        var statusSub = document.getElementById("market-time-status");
        var btnEl = document.getElementById("alert-btn");
        var isMarketHours = true;
        if ("Notification" in window && Notification.permission === "granted") {
            btnEl.style.display = "none";
            statusSub.innerText = isMarketHours
                ? "🟢 Market Open: Audio & Notifications ARMED."
                : "🌙 Market Closed: Standby.";
        } else {
            statusSub.innerText = "⚠️ Click once to enable browser sound permissions.";
        }
    }
    function activateSystemAlerts() {
        var statusEl = document.getElementById("alert-status-msg");
        if ("Notification" in window) {
            Notification.requestPermission().then(function(p) {
                if (p === "granted") { statusEl.innerText = "✅ Active!"; checkMarketHoursAndPermissions(); }
            });
        }
        try {
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                var ctx = new AudioCtx(); ctx.resume();
                var osc = ctx.createOscillator(); var gain = ctx.createGain();
                osc.type = "square"; osc.frequency.value = 900;
                osc.connect(gain); gain.connect(ctx.destination);
                gain.gain.setValueAtTime(1.0, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                osc.start(); osc.stop(ctx.currentTime + 0.5);
            }
        } catch(e) {}
    }
    window.onload = checkMarketHoursAndPermissions;
    setTimeout(checkMarketHoursAndPermissions, 1000);
    </script>
    """
    components.html(banner_html, height=85)


def play_trigger_alert(ticker, buy_price):
    js_html = f"""
    <script>
    (function() {{
        try {{
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {{
                var ctx = new AudioCtx(); ctx.resume();
                function beep(freq, start, dur) {{
                    var o = ctx.createOscillator(); var g = ctx.createGain();
                    o.type = "square"; o.frequency.value = freq;
                    o.connect(g); g.connect(ctx.destination);
                    g.gain.setValueAtTime(1.0, start);
                    g.gain.exponentialRampToValueAtTime(0.01, start + dur);
                    o.start(start); o.stop(start + dur);
                }}
                var now = ctx.currentTime;
                beep(900, now, 0.25); beep(900, now + 0.35, 0.25); beep(1200, now + 0.70, 0.5);
            }}
        }} catch(e) {{}}
        setTimeout(function() {{
            alert("🚨 PULLBACK ALERT: {ticker}\\n\\nTarget hit at ₹{buy_price:,.2f}. Trade moved to Paper Portfolio!");
        }}, 400);
    }})();
    </script>
    """
    components.html(js_html, height=0, width=0)


# ============================================================
# DATA HELPERS (cached, fast)
# ============================================================
@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_market_indices():
    items = [
        ("^NSEI", "Nifty 50", ""), ("^NSEBANK", "Bank Nifty", ""),
        ("^NSEMDCP50", "Nifty Midcap", ""), ("^CNXSC", "Nifty Smallcap", ""),
        ("^INDIAVIX", "India VIX", ""), ("CL=F", "Crude Oil", "$"),
    ]
    results = []
    try:
        data = yf.download(
            tickers=" ".join(t[0] for t in items), period="5d", interval="1d",
            group_by="ticker", threads=False, auto_adjust=True, progress=False, timeout=8,
        )
        for sym, name, prefix in items:
            try:
                if hasattr(data.columns, "levels") and sym in data.columns.levels[0]:
                    df = data[sym].dropna(how="all")
                else:
                    df = data.dropna(how="all")
                if df.empty:
                    continue
                curr = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr
                chg = curr - prev
                pct = (chg / prev * 100) if prev else 0
                results.append({
                    "name": name,
                    "value": f"{prefix}{curr:,.2f}" if prefix else f"{curr:,.2f}",
                    "change": f"{'+' if chg >= 0 else ''}{chg:.2f}",
                    "pct": f"({'+' if pct >= 0 else ''}{pct:.2f}%)",
                    "is_pos": chg >= 0,
                    "arrow": "↗" if chg >= 0 else "↘",
                })
            except Exception:
                continue
    except Exception:
        pass
    if not results:
        results = [{"name": n, "value": "—", "change": "—", "pct": "", "is_pos": True, "arrow": ""}
                   for n in ["Nifty 50", "Bank Nifty", "Nifty Midcap", "Nifty Smallcap", "India VIX", "Crude Oil"]]
    return results


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_ltp(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        p = getattr(t.fast_info, "last_price", None)
        if p and float(p) > 0 and not np.isnan(float(p)):
            return float(p)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            v = float(hist["Close"].iloc[-1])
            if v > 0 and not np.isnan(v):
                return v
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def get_real_fundamentals(ticker: str) -> dict:
    out = {"mcap_cr": 0.0, "roce": 0.0, "pe": 0.0, "de": 0.5, "pat_pct": 0.0, "pat_display": "—"}
    try:
        info = yf.Ticker(ticker).info or {}
        mcap = info.get("marketCap") or info.get("enterpriseValue") or 0
        if mcap and mcap > 0:
            out["mcap_cr"] = round(mcap / 1e7, 1)
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and isinstance(pe, (int, float)) and pe > 0:
            out["pe"] = round(float(pe), 1)
        de = info.get("debtToEquity")
        if de is not None:
            d = float(de)
            out["de"] = round(d / 100.0 if d > 5 else d, 2)
        roce = info.get("returnOnEquity") or info.get("returnOnAssets")
        if roce is not None:
            out["roce"] = round(float(roce) * 100, 1)
        gr = info.get("earningsQuarterlyGrowth") or info.get("earningsGrowth")
        if gr is not None:
            pct = float(gr) * 100
            out["pat_pct"] = pct
            out["pat_display"] = f"{pct:+.1f}%"
    except Exception:
        pass
    return out


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _parse_ai_verdict_to_score(text: str):
    t = (text or "").upper()
    if "STRONG BUY" in t:
        return 92, "🟢 STRONG BUY (AI)"
    if "BUY (ON PULLBACK)" in t or "BUY ON PULLBACK" in t:
        return 72, "🟡 BUY / PULLBACK (AI)"
    if "WAIT" in t:
        return 45, "🟠 WAIT (AI)"
    if "AVOID" in t or "SELL" in t:
        return 18, "🔴 AVOID (AI)"
    return 50, "🟠 CONSOLIDATING (AI)"


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


# ============================================================
# PERSISTENCE
# ============================================================
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
        st.error(f"Error saving {filename}: {e}")


# ============================================================
# ORDER BOOK + NSE UNIVERSE
# ============================================================
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
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS", "AUBANK.NS",
    ],
    "IT & Technology": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
        "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "KPITTECH.NS", "TATAELXSI.NS",
    ],
    "Automobile & EV": [
        "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "TVSMOTOR.NS", "EICHERMOT.NS", "BHARATFORG.NS", "SONACOMS.NS", "MOTHERSON.NS",
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "MANKIND.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS", "MAXHEALTH.NS",
    ],
    "Defence, Rail & PSUs": [
        "HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS", "RVNL.NS",
        "IRFC.NS", "COCHINSHIP.NS", "BDL.NS", "CONCOR.NS", "BEML.NS",
    ],
}


@st.cache_data(ttl=86400)
def get_all_nse_symbols():
    unique = sorted(list(dict.fromkeys(NSE_FULL_EQUITIES)))
    return [f"{s}.NS" for s in unique]


# ============================================================
# SCREENING ENGINE (optimized)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_screener_universe(ticker_list):
    if not ticker_list:
        return pd.DataFrame()
    unique_tickers = list(dict.fromkeys(ticker_list))
    total = len(unique_tickers)
    progress = st.progress(0, text="Fetching live market data...")
    chunk_size = 25 if total > 80 else 40
    chunks = [unique_tickers[i:i + chunk_size] for i in range(0, total, chunk_size)]
    rows, seen = [], set()

    for c_idx, chunk in enumerate(chunks):
        progress.progress((c_idx + 1) / len(chunks), text=f"Batch {c_idx+1}/{len(chunks)} ({min((c_idx+1)*chunk_size, total)}/{total})")
        batch = pd.DataFrame()
        for _ in range(2):
            try:
                batch = yf.download(
                    tickers=" ".join(chunk), period="6mo", interval="1d",
                    group_by="ticker", threads=True, auto_adjust=True, progress=False, timeout=12,
                )
                if not batch.empty:
                    break
            except Exception:
                time.sleep(0.6)
        if len(chunks) > 1:
            time.sleep(0.3)
        if batch.empty:
            continue

        for ticker in chunk:
            clean = ticker.replace(".NS", "").replace(".BO", "")
            if clean in seen:
                continue
            try:
                hist = pd.DataFrame()
                if isinstance(batch.columns, pd.MultiIndex):
                    if ticker in batch.columns.get_level_values(0):
                        hist = batch[ticker]
                elif len(chunk) == 1:
                    hist = batch
                hist = hist.dropna(how="all")
                if hist.empty or len(hist) < 26:
                    continue
                hist = hist[~hist.index.duplicated(keep="last")]

                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr
                chg_pct = round(((curr - prev) / prev) * 100, 2) if prev > 0 else 0.0

                ema9 = float(hist["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
                ema20 = float(hist["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema44 = float(hist["Close"].ewm(span=44, adjust=False).mean().iloc[-1])
                ema200 = float(hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1]) if len(hist) >= 200 else ema20
                sma50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else curr
                sma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else curr

                high52 = float(hist["High"].max())
                dist52 = max(0.0, ((high52 - curr) / high52) * 100)
                prev20h = float(hist["High"].iloc[:-1].tail(20).max()) if len(hist) > 20 else high52
                is_20d_bo = curr > prev20h

                rsi = compute_rsi(hist["Close"], 14)
                adx = compute_adx(hist, 14)

                # Weekly setup
                is_weekly = False
                weekly_df = pd.DataFrame()
                try:
                    tmp = hist.copy()
                    if getattr(tmp.index, "tz", None) is not None:
                        tmp.index = tmp.index.tz_convert(None)
                    weekly_df = tmp.resample("W-FRI").agg(
                        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                    ).dropna()
                    if len(weekly_df) >= 26:
                        wc = weekly_df["Close"]
                        wh, wl = weekly_df["High"], weekly_df["Low"]
                        e1 = wc.ewm(span=13, adjust=False).mean()
                        e2 = wc.ewm(span=21, adjust=False).mean()
                        macd = e1 - e2
                        sig = macd.ewm(span=9, adjust=False).mean()
                        macd_x = len(macd) >= 2 and macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2]
                        ll = wl.rolling(4).min(); hh = wh.rolling(4).max()
                        stoch = 100 * ((wc - ll) / (hh - ll + 1e-9))
                        stoch_x = len(stoch) >= 2 and stoch.iloc[-1] > 80 and stoch.iloc[-2] <= 80
                        dlt = wc.diff()
                        g = dlt.where(dlt > 0, 0.0); l = -dlt.where(dlt < 0, 0.0)
                        ag = g.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
                        al = l.ewm(alpha=1/7, min_periods=7, adjust=False).mean()
                        wrsi = 100 - (100 / (1 + ag / (al + 1e-9)))
                        rsi_x = len(wrsi) >= 2 and wrsi.iloc[-1] > 70 and wrsi.iloc[-2] <= 70
                        is_weekly = bool(macd_x and stoch_x and rsi_x)
                except Exception:
                    pass

                vol_s = hist["Volume"].dropna()
                curr_vol = int(vol_s.iloc[-1]) if not vol_s.empty else 0
                avg10 = float(vol_s.rolling(10).mean().iloc[-1]) if len(vol_s) >= 10 else float(curr_vol)
                avg20 = float(vol_s.rolling(20).mean().iloc[-1]) if len(vol_s) >= 20 else float(curr_vol)
                vol_surge = curr_vol >= avg20 * 0.95
                vol_2x = curr_vol > avg20 * 2.0

                cluster_hi = max(ema9, ema20, ema44, sma50)
                cluster_lo = min(ema9, ema20, ema44, sma50)
                spread = ((cluster_hi - cluster_lo) / cluster_hi * 100) if cluster_hi else 10
                is_squeeze = spread <= 4.5 and curr >= cluster_hi
                is_triple = ema9 > ema20 > ema44 and 30 <= curr <= 3000

                if len(weekly_df) >= 5:
                    w_c = float(weekly_df["Close"].iloc[-1])
                    w_e20 = float(weekly_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    w_rsi = compute_rsi(weekly_df["Close"], 14)
                    w_52 = float(weekly_df["High"].tail(52).max())
                else:
                    w_c, w_e20, w_rsi, w_52 = curr, ema20, rsi, high52
                is_mtf = (w_c > w_e20 and w_rsi >= 55 and curr > ema20 and is_20d_bo and vol_2x and curr >= w_52 * 0.80)

                c20 = float(hist["Close"].iloc[-21]) if len(hist) >= 21 else float(hist["Close"].iloc[0])
                c125 = float(hist["Close"].iloc[-126]) if len(hist) >= 126 else float(hist["Close"].iloc[0])
                is_rs = (
                    ((curr - ema200) / max(ema200, 1) * 100 > 30)
                    and ((curr - c125) / max(c125, 1) * 100 > 20)
                    and ((curr - sma50) / max(sma50, 1) * 100 > 20)
                    and ((curr - c20) / max(c20, 1) * 100 > 20)
                )

                is_overextended = chg_pct > 8.0 or rsi > 78
                score = 0
                if is_20d_bo: score += 25
                if vol_2x: score += 25
                if curr > ema9 > ema20: score += 25
                if adx >= 25: score += 25
                if chg_pct > 8: score -= 30
                swing = float(np.clip(score, 10, 100))

                if swing >= 80 and curr >= ema9 >= ema20 and not is_overextended:
                    signal = "🟢 STRONG BUY (Breakout)"
                elif (swing >= 50 or is_triple or is_squeeze or is_mtf or is_overextended) and curr >= ema20:
                    signal = "🟡 BUY / PULLBACK"
                elif swing >= 40:
                    signal = "🟠 CONSOLIDATING"
                else:
                    signal = "🔴 AVOID / WEAK"

                ob_val = ORDER_BOOK_CR_MAP.get(clean, 0.0)
                ob_disp = f"₹{ob_val:,.0f}" if ob_val > 0 else "—"

                rows.append({
                    "Ticker": clean, "Signal": signal,
                    "Price (₹)": round(curr, 2),
                    "Change (%)": f"{'+' if chg_pct >= 0 else ''}{chg_pct:.2f}%",
                    "Volume": f"{curr_vol:,}",
                    "Composite Score": round(swing, 1),
                    "ROCE (%)": 0.0, "PAT YoY (%)": "—",
                    "ADX (14)": adx, "RSI (14)": round(rsi, 1),
                    "From 52W High (%)": round(dist52, 1),
                    "Vol Surge": vol_surge,
                    "Market Cap (₹ Cr)": 0.0,
                    "Order Book (₹ Cr)": ob_disp, "OB / MCap": "—",
                    "9 EMA": round(ema9, 2), "20 EMA": round(ema20, 2), "44 EMA": round(ema44, 2),
                    "SMA_50": round(sma50, 2), "SMA_200": round(sma200, 2),
                    "Raw_Ticker": ticker,
                    "_raw_vol": curr_vol, "_avg_vol_10": avg10, "_avg_vol_20": avg20,
                    "_change_num": chg_pct, "_roce_num": 0.0, "_pat_num": 0.0,
                    "_de_num": 0.5, "_mcap_num": 0.0, "_adx_num": adx,
                    "_cluster_squeeze_match": is_squeeze, "_triple_ema_match": is_triple,
                    "_mtf_match": is_mtf, "_rs_match": is_rs,
                    "_ob_gt_mcap": False, "_weekly_setup_match": is_weekly,
                })
                seen.add(clean)
            except Exception:
                continue
        del batch
        gc.collect()

    progress.empty()
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def get_single_stock_history(ticker):
    try:
        t = ticker.strip()
        if not (t.endswith(".NS") or t.endswith(".BO")):
            t = f"{t}.NS"
        df = yf.download(tickers=t, period="1y", interval="1d", auto_adjust=True, progress=False, threads=False, timeout=8)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
            return df.dropna(how="all")
        return yf.Ticker(t).history(period="1y")
    except Exception:
        return pd.DataFrame()


# ============================================================
# SESSION STATE + SIDEBAR
# ============================================================
if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_json_file(PORTFOLIO_FILE)
if "pullback_watchlist" not in st.session_state:
    st.session_state["pullback_watchlist"] = load_json_file(WATCHLIST_FILE)
if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}
if "ai_score_map" not in st.session_state:
    st.session_state["ai_score_map"] = {}
if "screener_data" not in st.session_state:
    st.session_state["screener_data"] = pd.DataFrame()

WIDE_OPEN = {
    "sel_universe": "Nifty 50 Core", "scan_limit": 50, "strict_fund": False,
    "pat_growth": False, "ob_mcap": False, "roce_rng": (-20, 100),
    "mcap_rng": (0, 2000000), "max_de": 5.0, "price_rng": (10, 10000),
    "rsi_rng": (10, 95), "min_adx": 0, "dist_52w": 100, "ma_align": "Any Trend",
    "vol_10d_en": False, "vol_10d_mult": 1.1, "vol_20d_en": False, "vol_20d_mult": 1.2,
}
STRICT = {
    "sel_universe": "Nifty 50 Core", "scan_limit": 50, "strict_fund": True,
    "pat_growth": False, "ob_mcap": False, "roce_rng": (20, 100),
    "mcap_rng": (1000, 2000000), "max_de": 0.50, "price_rng": (30, 2000),
    "rsi_rng": (55, 75), "min_adx": 20, "dist_52w": 12, "ma_align": "Any Trend",
    "vol_10d_en": False, "vol_10d_mult": 1.1, "vol_20d_en": False, "vol_20d_mult": 1.2,
}
for k, v in WIDE_OPEN.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_open():
    for k, v in WIDE_OPEN.items():
        st.session_state[k] = v


def apply_strict():
    for k, v in STRICT.items():
        st.session_state[k] = v


# --- Live index ribbon ---
market_indices = fetch_live_market_indices()
if market_indices:
    html = '<div class="index-ticker-container">'
    for i, idx in enumerate(market_indices):
        cls = "index-pos" if idx["is_pos"] else "index-neg"
        html += f'<div class="index-item"><span class="index-name">{idx["name"]}</span><span class="index-val">{idx["value"]}</span><span class="{cls}">{idx["change"]} {idx["pct"]} {idx["arrow"]}</span></div>'
        if i < len(market_indices) - 1:
            html += '<span class="index-divider">|</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

st.title("⚡ Indian Market AI Stock Screener & Paper Trading")

# --- Sidebar ---
st.sidebar.header("🔑 API Setup")
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    GEMINI_API_KEY = str(api_key).strip()
    st.sidebar.success("✅ Gemini API Key connected")
else:
    GEMINI_API_KEY = st.sidebar.text_input("Google Gemini API Key", type="password")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY.strip())
    except Exception as e:
        st.sidebar.error(str(e))

selected_universe = st.sidebar.selectbox("Select Stock Basket", list(UNIVERSE_PRESETS.keys()), key="sel_universe")
is_single = selected_universe == "🔍 Single Stock Search"

if is_single:
    raw = st.sidebar.text_input("Enter NSE Symbol", value="ACE").strip().upper().replace(".NS", "").replace(".BO", "")
    tickers_to_scan = [f"{raw}.NS"] if raw else ["ACE.NS"]
elif selected_universe == "Nifty 50 Core":
    tickers_to_scan = get_all_nse_symbols()[:50]
elif selected_universe == "All NSE Stocks (Full Listed)":
    all_sym = get_all_nse_symbols()
    scan_limit = st.sidebar.slider("Number of Stocks to Scan", 25, min(300, len(all_sym)), step=25, key="scan_limit",
                                   help="Keep ≤ 100 for smooth live feel.")
    tickers_to_scan = all_sym[:scan_limit]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

st.sidebar.markdown("### Fundamental Filters")
apply_fund = st.sidebar.checkbox("Enable Strict Fundamental Filters", key="strict_fund")
pat_growth = st.sidebar.checkbox("PAT up > 20% YoY", key="pat_growth")
ob_mcap = st.sidebar.checkbox("Order Book > Market Cap", key="ob_mcap")
roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, key="roce_rng")
mcap_range = st.sidebar.slider("Market Cap (₹ Cr)", 0, 2000000, step=500, key="mcap_rng")
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, step=0.1, key="max_de")
price_range = st.sidebar.slider("Stock Price (₹)", 0, 10000, step=10, key="price_rng")
rsi_range = st.sidebar.slider("RSI (14)", 0, 100, key="rsi_rng")
min_adx = st.sidebar.slider("Min ADX", 0, 50, key="min_adx")
dist_52w = st.sidebar.slider("Within % of 52W High", 0, 100, key="dist_52w")
sma_filter = st.sidebar.selectbox("Moving Average Alignment", [
    "Any Trend", "🌀 EMA Cluster Squeeze & Breakout", "⚡ 9/20/44 Triple EMA Bullish Cross",
    "🔥 Multi-Timeframe 20D Breakout", "Relative strength", "Golden Cross (50 SMA > 200 SMA)",
    "⚡ Weekly MACD Crossover, Stochastics & RSI(7)",
], key="ma_align")
vol10_en = st.sidebar.checkbox("Volume > 10D SMA Multiplier", key="vol_10d_en")
vol10_m = st.sidebar.slider("10D Volume Surge Multiplier", 0.5, 5.0, step=0.1, disabled=not st.session_state.vol_10d_en, key="vol_10d_mult")
vol20_en = st.sidebar.checkbox("Volume > 20D SMA Multiplier", key="vol_20d_en")
vol20_m = st.sidebar.slider("20D Volume Surge Multiplier", 0.5, 5.0, step=0.1, disabled=not st.session_state.vol_20d_en, key="vol_20d_mult")

st.sidebar.button("🔓 Restore Open Defaults", on_click=reset_open, use_container_width=True)
st.sidebar.button("🎯 Apply Strict Strategy", on_click=apply_strict, use_container_width=True)
scan_btn = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)
if st.sidebar.button("🔄 Clear Cache & Rerun", use_container_width=True):
    st.cache_data.clear()
    st.session_state["ai_analysis_cache"] = {}
    st.session_state["ai_score_map"] = {}
    st.session_state["screener_data"] = pd.DataFrame()
    gc.collect()
    st.rerun()
st.sidebar.caption("⚡ Fast mode: Nifty 50 default. Scan ≤ 100 stocks. Fundamentals cached 1h.")

# --- Load / scan ---
_auto = is_single or selected_universe in (
    "Nifty 50 Core", "Banking & Financial Services", "IT & Technology",
    "Automobile & EV", "Pharma & Healthcare", "Defence, Rail & PSUs",
) or (selected_universe == "All NSE Stocks (Full Listed)" and st.session_state.get("scan_limit", 50) <= 60)

if st.session_state["screener_data"].empty and _auto:
    with st.spinner("Loading live data..."):
        st.session_state["screener_data"] = fetch_screener_universe(tickers_to_scan)

if scan_btn or is_single:
    with st.spinner("Refreshing live market data..."):
        st.session_state["screener_data"] = fetch_screener_universe(tickers_to_scan)

df_raw = st.session_state["screener_data"]

# --- Filter ---
filtered_df = pd.DataFrame()
if not df_raw.empty:
    filtered_df = df_raw.copy()
    if not is_single:
        filtered_df = filtered_df[
            (filtered_df["Price (₹)"] >= price_range[0]) & (filtered_df["Price (₹)"] <= price_range[1])
            & (filtered_df["RSI (14)"] >= rsi_range[0]) & (filtered_df["RSI (14)"] <= rsi_range[1])
            & (filtered_df["_adx_num"] >= min_adx) & (filtered_df["From 52W High (%)"] <= dist_52w)
        ]
        if sma_filter == "🌀 EMA Cluster Squeeze & Breakout":
            filtered_df = filtered_df[filtered_df["_cluster_squeeze_match"]]
        elif sma_filter == "⚡ 9/20/44 Triple EMA Bullish Cross":
            filtered_df = filtered_df[filtered_df["_triple_ema_match"]]
        elif sma_filter == "🔥 Multi-Timeframe 20D Breakout":
            filtered_df = filtered_df[filtered_df["_mtf_match"]]
        elif sma_filter == "Relative strength":
            filtered_df = filtered_df[filtered_df["_rs_match"]]
        elif sma_filter == "Golden Cross (50 SMA > 200 SMA)":
            filtered_df = filtered_df[filtered_df["SMA_50"] >= filtered_df["SMA_200"]]
        elif sma_filter == "⚡ Weekly MACD Crossover, Stochastics & RSI(7)":
            filtered_df = filtered_df[filtered_df["_weekly_setup_match"]]
        if vol10_en:
            filtered_df = filtered_df[filtered_df["_raw_vol"] >= filtered_df["_avg_vol_10"] * vol10_m]
        if vol20_en:
            filtered_df = filtered_df[filtered_df["_raw_vol"] >= filtered_df["_avg_vol_20"] * vol20_m]

    # Real fundamentals only when PAT filter ON (keeps UI fast)
    if not filtered_df.empty and pat_growth:
        filtered_df = filtered_df.copy()
        if len(filtered_df) > 20:
            filtered_df = filtered_df.nlargest(20, "_raw_vol")
        with st.spinner("Fetching PAT YoY..."):
            keep = []
            for idx, row in filtered_df.iterrows():
                try:
                    fund = get_real_fundamentals(row["Raw_Ticker"])
                    filtered_df.at[idx, "ROCE (%)"] = fund["roce"]
                    filtered_df.at[idx, "_roce_num"] = fund["roce"]
                    filtered_df.at[idx, "Market Cap (₹ Cr)"] = fund["mcap_cr"]
                    filtered_df.at[idx, "_mcap_num"] = fund["mcap_cr"]
                    filtered_df.at[idx, "_de_num"] = fund["de"]
                    filtered_df.at[idx, "PAT YoY (%)"] = fund["pat_display"]
                    filtered_df.at[idx, "_pat_num"] = fund["pat_pct"]
                    if fund["pat_pct"] >= 20:
                        keep.append(idx)
                except Exception:
                    pass
            filtered_df = filtered_df.loc[keep] if keep else filtered_df.iloc[0:0]
    elif not filtered_df.empty and apply_fund:
        # Light fund filter using placeholders; real fetch on demand in deepdive
        pass

# Apply AI-adjusted scores to filtered view
if not filtered_df.empty and st.session_state.get("ai_score_map"):
    for tkr, (sc, sig) in st.session_state["ai_score_map"].items():
        mask = filtered_df["Raw_Ticker"] == tkr
        if mask.any():
            filtered_df.loc[mask, "Composite Score"] = sc
            filtered_df.loc[mask, "Signal"] = sig

# ============================================================
# TABS
# ============================================================
tab_screener, tab_deepdive, tab_pullback, tab_portfolio, tab_rebalance = st.tabs([
    "📊 Screener & Momentum Signals",
    "🔬 Single Stock Chart & AI Thesis",
    "🎯 Pullback Watchlist & Order Trigger",
    "💼 Paper Trading Portfolio",
    "⚖️ Portfolio Rebalance",
])

# ---------- SCREENER ----------
with tab_screener:
    if df_raw.empty:
        st.warning("No stocks fetched. Reduce scan size or click Run Screener Scan.")
    elif filtered_df.empty:
        st.info("0 matching stocks. Relax filters.")
    else:
        c1, c2, c3 = st.columns([2, 1.2, 1])
        with c1:
            st.subheader(f"Matching Stocks ({len(filtered_df)} of {len(df_raw)})")
        with c2:
            sort_m = st.selectbox("Sort By:", ["Volume", "Composite Score", "Change (%)", "Price (₹)", "ADX (14)", "RSI (14)", "From 52W High (%)"])
        with c3:
            sort_dir = st.radio("Order:", ["High→Low", "Low→High"], horizontal=True)
        col_map = {"Volume": "_raw_vol", "Composite Score": "Composite Score", "Change (%)": "_change_num",
                   "Price (₹)": "Price (₹)", "ADX (14)": "_adx_num", "RSI (14)": "RSI (14)", "From 52W High (%)": "From 52W High (%)"}
        sorted_df = filtered_df.sort_values(by=col_map.get(sort_m, "_raw_vol"), ascending=(sort_dir == "Low→High"), na_position="last")
        cols = ["Ticker", "Signal", "Price (₹)", "Change (%)", "Volume", "Composite Score", "ROCE (%)", "PAT YoY (%)",
                "ADX (14)", "RSI (14)", "From 52W High (%)", "Vol Surge", "Market Cap (₹ Cr)", "Order Book (₹ Cr)", "OB / MCap"]
        table = sorted_df[cols].copy()
        table["Price (₹)"] = table["Price (₹)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "-")
        table["Composite Score"] = table["Composite Score"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        table["Vol Surge"] = table["Vol Surge"].apply(lambda x: "✅" if x else "⬜")
        sel = st.dataframe(table, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        if sel and sel.selection and sel.selection.rows:
            st.session_state["selected_ticker"] = f"{table.iloc[sel.selection.rows[0]]['Ticker']}.NS"

# ---------- DEEP DIVE + AI ----------
with tab_deepdive:
    if df_raw.empty:
        st.write("No data.")
    else:
        opts = filtered_df["Raw_Ticker"].tolist() if not filtered_df.empty else df_raw["Raw_Ticker"].tolist()
        cur = st.session_state.get("selected_ticker", opts[0] if opts else "ACE.NS")
        idx = opts.index(cur) if cur in opts else 0
        selected = st.selectbox("Selected Stock:", opts, index=idx)
        st.session_state["selected_ticker"] = selected

        hist = get_single_stock_history(selected)
        match = df_raw[df_raw["Raw_Ticker"] == selected]
        row = match.iloc[0] if not match.empty else None

        if hist is not None and not hist.empty:
            hist["EMA_9"] = hist["Close"].ewm(span=9, adjust=False).mean()
            hist["EMA_20"] = hist["Close"].ewm(span=20, adjust=False).mean()
            hist["EMA_44"] = hist["Close"].ewm(span=44, adjust=False).mean()
            hist["SMA_50"] = hist["Close"].rolling(50).mean()
            hist["SMA_200"] = hist["Close"].rolling(200).mean()
            cp = float(hist["Close"].iloc[-1])
            e9, e20, e44 = float(hist["EMA_9"].iloc[-1]), float(hist["EMA_20"].iloc[-1]), float(hist["EMA_44"].iloc[-1])
            signal = row["Signal"] if row is not None else "N/A"
            score = row["Composite Score"] if row is not None else 0
            adx = row["ADX (14)"] if row is not None else 25
            chg = row["Change (%)"] if row is not None else "0%"
            vol = row["Volume"] if row is not None else "—"
            if selected in st.session_state.get("ai_score_map", {}):
                score, signal = st.session_state["ai_score_map"][selected]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"₹{cp:,.2f}", delta=chg)
            m2.metric("9 / 20 / 44 EMA", f"₹{e9:.1f} / ₹{e20:.1f} / ₹{e44:.1f}")
            m3.metric("Volume / ADX", f"{vol} | ADX: {adx}")
            m4.metric("Score / Signal", f"{int(score)}/100", delta=signal)

            fig = go.Figure(data=[
                go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"),
                go.Scatter(x=hist.index, y=hist["EMA_9"], line=dict(color="#00f2ff", width=1.5), name="9 EMA"),
                go.Scatter(x=hist.index, y=hist["EMA_20"], line=dict(color="#ffd700", width=1.5), name="20 EMA"),
                go.Scatter(x=hist.index, y=hist["EMA_44"], line=dict(color="#a855f7", width=1.5), name="44 EMA"),
                go.Scatter(x=hist.index, y=hist["SMA_50"], line=dict(color="#ff9900", width=1.5), name="50 SMA"),
                go.Scatter(x=hist.index, y=hist["SMA_200"], line=dict(color="#4d79ff", width=1.5), name="200 SMA"),
            ])
            fig.update_layout(template="plotly_dark", height=480, margin=dict(l=20, r=20, t=30, b=20),
                              xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🤖 AI Short-Term Swing Thesis")
            cached = st.session_state["ai_analysis_cache"].get(selected)
            if cached:
                st.markdown(cached)
                if selected in st.session_state["ai_score_map"]:
                    sc, sg = st.session_state["ai_score_map"][selected]
                    st.success(f"AI-Aligned Score: **{sc}/100** → {sg}")

            if st.button(f"Generate Short-Term Swing Setup for {selected}"):
                if not GEMINI_API_KEY:
                    st.warning("Add Gemini API Key in sidebar.")
                else:
                    prompt = f"""NSE swing trader. Be concise (max 120 words).
Stock: {selected} | Price: ₹{cp:.2f} ({chg}) | Vol: {vol}
EMA 9/20/44: {e9:.1f}/{e20:.1f}/{e44:.1f} | ADX: {adx} | RSI: {row['RSI (14)'] if row is not None else 'N/A'} | Score: {score}

Reply EXACTLY:
VERDICT: <STRONG BUY | BUY (ON PULLBACK) | WAIT | AVOID>
Setup: <1 line>
Entry: ₹x–y | SL: ₹z | T1: ₹a | T2: ₹b
Exit: <1 line>"""
                    with st.spinner("AI analyzing (fast)..."):
                        ok, errs = False, []
                        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"]:
                            try:
                                model = genai.GenerativeModel(model_name)
                                res = model.generate_content(prompt, generation_config={"max_output_tokens": 220, "temperature": 0.3})
                                if res and res.text:
                                    txt = res.text.strip()
                                    st.session_state["ai_analysis_cache"][selected] = txt
                                    st.markdown(txt)
                                    vs, vsig = _parse_ai_verdict_to_score(txt)
                                    st.session_state["ai_score_map"][selected] = (vs, vsig)
                                    if not st.session_state["screener_data"].empty:
                                        mk = st.session_state["screener_data"]["Raw_Ticker"] == selected
                                        if mk.any():
                                            st.session_state["screener_data"].loc[mk, "Composite Score"] = vs
                                            st.session_state["screener_data"].loc[mk, "Signal"] = vsig
                                    st.success(f"AI Score: {vs}/100 → {vsig}")
                                    ok = True
                                    break
                            except Exception as e:
                                errs.append(f"{model_name}: {e}")
                        if not ok:
                            st.error("AI failed.")
                            with st.expander("Errors"):
                                for e in errs:
                                    st.code(e)

# ---------- PULLBACK WATCHLIST ----------
with tab_pullback:
    st.subheader("🎯 Pullback Watchlist & Limit Order Execution")
    render_alert_permission_banner()
    st.info("Place limit orders below LTP. When price ≤ target, alert fires and trade moves to Paper Portfolio.")

    c_dl, c_up = st.columns(2)
    with c_up:
        up = st.file_uploader("📥 Restore Watchlist (.json)", type=["json"], key="wl_up")
        if up:
            try:
                data = json.load(up)
                if isinstance(data, list):
                    st.session_state["pullback_watchlist"] = [x for x in data if isinstance(x, dict) and "Target Buy (₹)" in x]
                    save_json_file(WATCHLIST_FILE, st.session_state["pullback_watchlist"])
                    st.success("Restored!"); st.rerun()
            except Exception as e:
                st.error(str(e))
    with c_dl:
        if st.session_state.get("pullback_watchlist"):
            st.download_button("💾 Download Watchlist", data=json.dumps(st.session_state["pullback_watchlist"], indent=4),
                               file_name="watchlist_backup.json", mime="application/json", use_container_width=True)

    if not df_raw.empty:
        cands = df_raw["Raw_Ticker"].tolist()
        cur = st.session_state.get("selected_ticker", cands[0])
        di = cands.index(cur) if cur in cands else 0
        with st.expander("➕ Add to Pullback Watchlist", expanded=False):
            with st.form("add_wl_form"):
                a1, a2, a3, a4, a5 = st.columns(5)
                with a1:
                    sel = st.selectbox("Stock", cands, index=di)
                    mr = df_raw[df_raw["Raw_Ticker"] == sel]
                    ltp = float(mr["Price (₹)"].iloc[0]) if not mr.empty else 100.0
                    e20 = float(mr["20 EMA"].iloc[0]) if not mr.empty else ltp * 0.98
                with a2:
                    st.metric("LTP", f"₹{ltp:,.2f}")
                with a3:
                    tgt_buy = st.number_input("Target Entry ₹", value=round(e20, 2), min_value=0.1, step=0.5)
                with a4:
                    slv = st.number_input("SL ₹", value=round(tgt_buy * 0.95, 2), min_value=0.0, step=0.5)
                with a5:
                    tgv = st.number_input("TGT ₹", value=round(tgt_buy * 1.10, 2), min_value=0.0, step=0.5)
                q = st.number_input("Qty", value=50, min_value=1, step=1)
                note = st.text_input("Strategy Note", value=sma_filter)
                if st.form_submit_button("📥 Add to Watchlist", use_container_width=True):
                    clean = sel.replace(".NS", "").replace(".BO", "")
                    item = {
                        "id": f"wb_{clean}_{int(time.time())}", "Date Added": str(date.today()),
                        "Ticker": clean, "Raw_Ticker": sel, "Target Buy (₹)": float(tgt_buy),
                        "SL (₹)": float(slv), "TGT (₹)": float(tgv), "Qty": int(q),
                        "Strategy": note.strip(), "Status": "⏳ Waiting for Pullback",
                    }
                    st.session_state["pullback_watchlist"].append(item)
                    save_json_file(WATCHLIST_FILE, st.session_state["pullback_watchlist"])
                    st.success(f"Added {clean}"); st.rerun()

    wl = st.session_state.get("pullback_watchlist", [])
    if wl:
        live = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"])) if not df_raw.empty else {}
        updated, display = [], []
        for item in wl:
            if not isinstance(item, dict) or "Target Buy (₹)" not in item:
                continue
            sym = item.get("Raw_Ticker") or f"{item.get('Ticker')}.NS"
            clean = item.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
            tb = _safe_float(item.get("Target Buy (₹)"))
            slp = _safe_float(item.get("SL (₹)"))
            tgp = _safe_float(item.get("TGT (₹)"))
            qty = int(item.get("Qty", 1) or 1)
            status = item.get("Status", "⏳ Waiting for Pullback")
            ltp = live.get(sym)
            if ltp is None:
                ltp = get_cached_ltp(sym) or None
            else:
                ltp = _safe_float(ltp, None)

            if "Waiting" in status and ltp and ltp > 0 and ltp <= tb:
                status = "⚡ Triggered / Bought"
                item["Status"] = status
                play_trigger_alert(clean, tb)
                st.toast(f"🎯 {clean} bought @ ₹{ltp:.2f}!", icon="⚡")
                trade = {
                    "id": f"{sym}_{int(time.time())}", "Date": str(date.today()), "Exit_Date": "",
                    "Ticker": clean, "Buy Price (₹)": ltp, "SL (₹)": slp, "TGT (₹)": tgp,
                    "Exit Price (₹)": 0.0, "Qty": qty,
                    "Remarks": f"Pullback Auto-Entry ({item.get('Strategy', '')})",
                    "Status": "🟢 Open", "Invested (₹)": round(ltp * qty, 2), "Raw_Ticker": sym,
                }
                if not any(p.get("id") == trade["id"] for p in st.session_state["paper_portfolio"]):
                    st.session_state["paper_portfolio"].append(trade)
                    save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])

            item["Status"] = status
            updated.append(item)
            dist = f"{((ltp - tb) / tb * 100):+.2f}% away" if (ltp and tb > 0 and "Waiting" in status) else ("Executed ✅" if "Triggered" in status else "—")
            display.append({
                "Date Added": item.get("Date Added", ""), "Ticker": clean,
                "LTP (₹)": f"₹{ltp:,.2f}" if ltp else "—", "Target Buy (₹)": f"₹{tb:,.2f}",
                "Distance": dist, "SL (₹)": f"₹{slp:,.2f}", "TGT (₹)": f"₹{tgp:,.2f}",
                "Qty": qty, "Status": status, "Strategy": item.get("Strategy", ""),
            })

        if updated != st.session_state.get("pullback_watchlist"):
            st.session_state["pullback_watchlist"] = updated
            save_json_file(WATCHLIST_FILE, updated)
        else:
            st.session_state["pullback_watchlist"] = updated
        if display:
            st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

        with st.expander("✏️ Edit Watchlist Item", expanded=False):
            opts = {f"{it.get('Ticker')} (₹{it.get('Target Buy (₹)')}) [{it.get('Status')}]": i for i, it in enumerate(updated)}
            if opts:
                with st.form("edit_wl_form"):
                    lab = st.selectbox("Select", list(opts.keys()))
                    i = opts[lab]; cur = updated[i]
                    e1, e2, e3, e4, e5 = st.columns(5)
                    with e1: nt = st.number_input("Target Entry", value=_safe_float(cur.get("Target Buy (₹)")), step=0.5)
                    with e2: ns = st.number_input("SL", value=_safe_float(cur.get("SL (₹)")), step=0.5)
                    with e3: ng = st.number_input("TGT", value=_safe_float(cur.get("TGT (₹)")), step=0.5)
                    with e4: nq = st.number_input("Qty", value=int(cur.get("Qty", 50)), min_value=1, step=1)
                    with e5: nn = st.text_input("Note", value=cur.get("Strategy", ""))
                    if st.form_submit_button("💾 Save", use_container_width=True):
                        updated[i].update({"Target Buy (₹)": nt, "SL (₹)": ns, "TGT (₹)": ng, "Qty": nq, "Strategy": nn.strip()})
                        st.session_state["pullback_watchlist"] = updated
                        save_json_file(WATCHLIST_FILE, updated)
                        st.success("Saved"); st.rerun()

        d1, d2, d3 = st.columns([2, 1, 1])
        with d1:
            del_opts = {f"{it.get('Ticker')} [{it.get('Status')}]": i for i, it in enumerate(updated)}
            if del_opts:
                sel_del = st.selectbox("Manage item", list(del_opts.keys()), key="wl_del_sel")
        with d2:
            if del_opts and st.button("🔄 Re-Arm", use_container_width=True):
                i = del_opts[sel_del]
                sym = updated[i].get("Ticker")
                updated[i]["Status"] = "⏳ Waiting for Pullback"
                st.session_state["pullback_watchlist"] = updated
                save_json_file(WATCHLIST_FILE, updated)
                st.session_state["paper_portfolio"] = [p for p in st.session_state["paper_portfolio"] if not (p.get("Ticker") == sym and "Pullback" in p.get("Remarks", ""))]
                save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                st.rerun()
        with d3:
            if del_opts and st.button("🗑️ Delete", type="primary", use_container_width=True):
                i = del_opts[sel_del]
                sym = updated[i].get("Ticker")
                updated.pop(i)
                st.session_state["pullback_watchlist"] = updated
                save_json_file(WATCHLIST_FILE, updated)
                st.session_state["paper_portfolio"] = [p for p in st.session_state["paper_portfolio"] if not (p.get("Ticker") == sym and "Pullback" in p.get("Remarks", ""))]
                save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                st.rerun()
    else:
        st.info("Watchlist empty. Add a setup above.")

# ---------- PAPER PORTFOLIO ----------
with tab_portfolio:
    st.subheader("💼 Paper Trading Portfolio & Risk Manager")
    port = st.session_state.get("paper_portfolio", [])

    if not df_raw.empty:
        with st.expander("➕ Execute New Paper Trade", expanded=False):
            with st.form("new_trade_form"):
                avail = df_raw["Raw_Ticker"].tolist()
                cur = st.session_state.get("selected_ticker", avail[0])
                di = avail.index(cur) if cur in avail else 0
                t1, t2, t3, t4, t5 = st.columns(5)
                with t1: ts = st.selectbox("Stock", avail, index=di)
                with t2: td = st.date_input("Entry Date", value=date.today())
                with t3:
                    ms = df_raw[df_raw["Raw_Ticker"] == ts]
                    lp = float(ms["Price (₹)"].iloc[0]) if not ms.empty else 100.0
                    bp = st.number_input("Entry ₹", value=lp, min_value=0.1, step=0.5)
                with t4: sl = st.number_input("SL ₹", value=round(bp * 0.96, 1), min_value=0.0, step=0.5)
                with t5: tg = st.number_input("TGT ₹", value=round(bp * 1.08, 1), min_value=0.0, step=0.5)
                qty = st.number_input("Qty", value=50, min_value=1, step=1)
                rem = st.text_input("Remarks", value=sma_filter)
                if st.form_submit_button("📥 Execute Trade", use_container_width=True):
                    raw = ts if ts.endswith(".NS") else f"{ts}.NS"
                    trade = {
                        "id": f"{raw}_{int(time.time())}", "Date": str(td), "Exit_Date": "",
                        "Ticker": raw.replace(".NS", "").replace(".BO", ""),
                        "Buy Price (₹)": bp, "SL (₹)": sl, "TGT (₹)": tg, "Exit Price (₹)": 0.0,
                        "Qty": int(qty), "Remarks": rem.strip(), "Status": "🟢 Open",
                        "Invested (₹)": round(bp * qty, 2), "Raw_Ticker": raw,
                    }
                    st.session_state["paper_portfolio"].append(trade)
                    save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                    st.success("Trade executed"); st.rerun()

    if port:
        with st.expander("🗑️ Delete a Trade", expanded=False):
            opts = {f"{p.get('Ticker')} (₹{p.get('Buy Price (₹)')} on {p.get('Date')}) [{p.get('Status')}]": i for i, p in enumerate(port)}
            if opts:
                lab = st.selectbox("Select to delete", list(opts.keys()), key="del_trade")
                if st.button("🗑️ Delete Selected", type="primary"):
                    port.pop(opts[lab])
                    st.session_state["paper_portfolio"] = port
                    save_json_file(PORTFOLIO_FILE, port)
                    st.success("Deleted"); st.rerun()

    c1, c2 = st.columns(2)
    with c2:
        up = st.file_uploader("📥 Restore Portfolio (.json)", type=["json"], key="port_up")
        if up:
            try:
                data = json.load(up)
                if isinstance(data, list) and data:
                    st.session_state["paper_portfolio"] = data
                    save_json_file(PORTFOLIO_FILE, data)
                    st.success("Restored!")
            except Exception as e:
                st.error(str(e))
    with c1:
        if port:
            st.download_button("💾 Download Portfolio", data=json.dumps(port, indent=4),
                               file_name="portfolio_backup.json", mime="application/json", use_container_width=True)

    if port:
        live = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"])) if not df_raw.empty else {}
        open_inv = open_val = u_pnl = r_pnl = 0.0
        wins = losses = opens = 0
        total = len(port)
        prows, updated = [], []

        for pos in port:
            sym = pos.get("Raw_Ticker", f"{pos.get('Ticker', 'ACE')}.NS")
            clean = pos.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
            buy = _safe_float(pos.get("Buy Price (₹)"))
            qty = max(1, int(pos.get("Qty", 1) or 1))
            inv = _safe_float(pos.get("Invested (₹)"), buy * qty)
            if inv <= 0:
                inv = buy * qty
            sl = _safe_float(pos.get("SL (₹)"))
            tgt = _safe_float(pos.get("TGT (₹)"))
            status = str(pos.get("Status", "🟢 Open"))
            exit_p = _safe_float(pos.get("Exit Price (₹)"))
            exit_d = str(pos.get("Exit_Date", "") or "")
            entry_d = str(pos.get("Date", date.today()))

            curr = live.get(sym)
            curr = _safe_float(curr, None)
            if curr is None or curr <= 0:
                curr = get_cached_ltp(sym)
            if curr <= 0:
                curr = buy if buy > 0 else 0.0

            if status == "🟢 Open" and curr > 0:
                if sl > 0 and curr <= sl:
                    status, exit_d, exit_p = "🔴 SL Hit (Closed)", str(date.today()), sl
                elif tgt > 0 and curr >= tgt:
                    status, exit_d, exit_p = "🎯 TGT Hit (Closed)", str(date.today()), tgt

            pos["Status"] = status
            pos["Exit_Date"] = exit_d
            pos["Exit Price (₹)"] = exit_p
            updated.append(pos)

            try:
                de = datetime.strptime(entry_d, "%Y-%m-%d").date()
                dx = datetime.strptime(exit_d, "%Y-%m-%d").date() if exit_d and exit_d != "-" else date.today()
                hold = max(0, (dx - de).days)
            except Exception:
                hold = 0

            if "Closed" in status or status == "⚪ Sold Manually":
                ev = exit_p if exit_p > 0 else (curr if curr > 0 else buy)
                pnl = round((ev - buy) * qty, 2)
                r_pnl += pnl
                eff = ev
            else:
                opens += 1
                pnl = round((curr - buy) * qty, 2)
                open_inv += inv
                open_val += round(curr * qty, 2)
                u_pnl += pnl
                eff = curr

            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1
            pct = round(pnl / inv * 100, 2) if inv > 0 else 0.0

            prows.append({
                "Entry Date": entry_d, "Sold Date": exit_d or "—", "Holding": f"{hold} d",
                "Ticker": clean, "Status": status, "Remarks": pos.get("Remarks", ""),
                "Entry (₹)": f"₹{buy:,.2f}", "SL (₹)": f"₹{sl:,.2f}" if sl else "—",
                "TGT (₹)": f"₹{tgt:,.2f}" if tgt else "—",
                "Current Price (₹)": f"₹{eff:,.2f}", "Qty": qty,
                "Invested (₹)": f"₹{inv:,.2f}",
                "P&L (₹)": f"{'+' if pnl >= 0 else ''}₹{pnl:,.2f}",
                "P&L (%)": f"{'+' if pct >= 0 else ''}{pct:.2f}%",
            })

        # Summary cards
        win_r = wins / total * 100 if total else 0
        loss_r = losses / total * 100 if total else 0
        st.markdown(f"""
        <div class="trade-summary-card">
            <div class="trade-stat-box"><div class="trade-stat-label">Total Trades</div><div class="trade-stat-val">{total}</div></div>
            <div class="trade-stat-box"><div class="trade-stat-label">Winning</div><div class="trade-stat-val" style="color:#16a34a">{wins}</div></div>
            <div class="trade-stat-box"><div class="trade-stat-label">Losing</div><div class="trade-stat-val" style="color:#dc2626">{losses}</div></div>
            <div class="trade-stat-box"><div class="trade-stat-label">Open</div><div class="trade-stat-val" style="color:#0284c7">{opens}</div></div>
        </div>
        """, unsafe_allow_html=True)

        if np.isnan(open_inv): open_inv = 0.0
        if np.isnan(open_val): open_val = open_inv
        if np.isnan(u_pnl): u_pnl = 0.0
        if np.isnan(r_pnl): r_pnl = 0.0
        upct = (u_pnl / open_inv * 100) if open_inv > 0 else 0.0

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Open Invested", f"₹{open_inv:,.2f}")
        p2.metric("Open Portfolio Value", f"₹{open_val:,.2f}")
        p3.metric("Unrealised P&L", f"₹{u_pnl:,.2f}", delta=f"{upct:.2f}%")
        p4.metric("Realised P&L", f"₹{r_pnl:,.2f}")
        p5.metric("Win / Loss %", f"{win_r:.1f}% Win", delta=f"{loss_r:.1f}% Loss", delta_color="inverse")

        with st.expander("✏️ Edit Position (form = no lag)", expanded=False):
            eopts = {f"{p.get('Ticker')} (₹{p.get('Buy Price (₹)')} · {p.get('Status')})": i for i, p in enumerate(port)}
            if eopts:
                with st.form("edit_pos_form"):
                    lab = st.selectbox("Position", list(eopts.keys()))
                    i = eopts[lab]; cur = port[i]
                    e1, e2, e3, e4, e5 = st.columns(5)
                    with e1: nsl = st.number_input("SL ₹", value=_safe_float(cur.get("SL (₹)")), step=0.5)
                    with e2: ntg = st.number_input("TGT ₹", value=_safe_float(cur.get("TGT (₹)")), step=0.5)
                    with e3:
                        opts_s = ["🟢 Open", "🔴 SL Hit (Closed)", "🎯 TGT Hit (Closed)", "⚪ Sold Manually"]
                        si = opts_s.index(cur.get("Status", "🟢 Open")) if cur.get("Status") in opts_s else 0
                        nst = st.selectbox("Status", opts_s, index=si)
                    with e4:
                        try:
                            ed = datetime.strptime(str(cur.get("Exit_Date") or date.today()), "%Y-%m-%d").date()
                        except Exception:
                            ed = date.today()
                        ned = st.date_input("Sold Date", value=ed)
                    with e5: nep = st.number_input("Exit ₹", value=_safe_float(cur.get("Exit Price (₹)"), _safe_float(cur.get("Buy Price (₹)"))), step=0.5)
                    if st.form_submit_button("💾 Save", use_container_width=True):
                        port[i]["SL (₹)"] = nsl
                        port[i]["TGT (₹)"] = ntg
                        port[i]["Status"] = nst
                        if nst != "🟢 Open":
                            port[i]["Exit_Date"] = str(ned)
                            port[i]["Exit Price (₹)"] = nep
                        else:
                            port[i]["Exit_Date"] = ""
                            port[i]["Exit Price (₹)"] = 0.0
                        save_json_file(PORTFOLIO_FILE, port)
                        st.success("Updated"); st.rerun()

        pdf = pd.DataFrame(prows)
        st.dataframe(pdf, use_container_width=True, hide_index=True)
        if st.button("🗑️ Reset / Clear All Trades"):
            st.session_state["paper_portfolio"] = []
            save_json_file(PORTFOLIO_FILE, [])
            st.rerun()

# ---------- REBALANCE (with Edit + Delete) ----------
with tab_rebalance:
    st.subheader("⚖️ Automated Portfolio Rebalancing")
    st.caption("Open positions only. Includes Edit & Delete for individual stocks.")

    portfolio = st.session_state.get("paper_portfolio", [])
    open_pos = [p for p in portfolio if str(p.get("Status", "")).startswith("🟢")]

    if not open_pos:
        st.info("No open positions. Add trades in Paper Trading first.")
    else:
        live = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"])) if not df_raw.empty else {}
        rows = []
        total_val = 0.0
        for pos in open_pos:
            sym = pos.get("Raw_Ticker", f"{pos.get('Ticker', 'ACE')}.NS")
            clean = pos.get("Ticker", sym.replace(".NS", "").replace(".BO", ""))
            buy = _safe_float(pos.get("Buy Price (₹)"))
            qty = max(1, int(pos.get("Qty", 1) or 1))
            curr = live.get(sym)
            curr = _safe_float(curr, None)
            if curr is None or curr <= 0:
                curr = get_cached_ltp(sym)
            if curr <= 0:
                curr = buy if buy > 0 else 0.0
            val = round(curr * qty, 2)
            total_val += val
            rows.append({
                "id": pos.get("id"), "Ticker": clean, "Raw_Ticker": sym,
                "Qty": qty, "Buy (₹)": buy, "LTP (₹)": round(curr, 2),
                "Value (₹)": val, "SL (₹)": _safe_float(pos.get("SL (₹)")),
                "TGT (₹)": _safe_float(pos.get("TGT (₹)")),
                "Status": pos.get("Status", "🟢 Open"),
            })

        if total_val <= 0:
            st.warning("Could not price portfolio. Run screener first.")
        else:
            for r in rows:
                r["Weight %"] = round(r["Value (₹)"] / total_val * 100, 2)
            st.markdown(f"**Open:** {len(rows)} · **Value:** ₹{total_val:,.2f}")

            # --- Edit / Delete individual stocks ---
            st.markdown("#### ✏️ Edit or 🗑️ Delete a Position")
            rb_opts = {f"{r['Ticker']} · Qty {r['Qty']} · ₹{r['LTP (₹)']:,.2f} ({r['Weight %']:.1f}%)": r["id"] for r in rows}
            if rb_opts:
                with st.form("rebalance_edit_form"):
                    sel_lab = st.selectbox("Select stock", list(rb_opts.keys()))
                    sel_id = rb_opts[sel_lab]
                    # find position in full portfolio
                    pos_idx = next((i for i, p in enumerate(portfolio) if p.get("id") == sel_id), None)
                    if pos_idx is not None:
                        cur = portfolio[pos_idx]
                        e1, e2, e3, e4 = st.columns(4)
                        with e1:
                            new_qty = st.number_input("Qty", value=max(1, int(cur.get("Qty", 1))), min_value=0, step=1)
                        with e2:
                            new_sl = st.number_input("SL ₹", value=_safe_float(cur.get("SL (₹)")), step=0.5)
                        with e3:
                            new_tgt = st.number_input("TGT ₹", value=_safe_float(cur.get("TGT (₹)")), step=0.5)
                        with e4:
                            new_buy = st.number_input("Buy Price ₹", value=_safe_float(cur.get("Buy Price (₹)")), min_value=0.0, step=0.5)
                        c_save, c_del = st.columns(2)
                        with c_save:
                            do_save = st.form_submit_button("💾 Save Changes", use_container_width=True)
                        with c_del:
                            do_del = st.form_submit_button("🗑️ Delete Position", use_container_width=True, type="primary")
                        if do_save and pos_idx is not None:
                            if new_qty <= 0:
                                portfolio[pos_idx]["Status"] = "⚪ Sold Manually"
                                portfolio[pos_idx]["Exit_Date"] = str(date.today())
                                portfolio[pos_idx]["Exit Price (₹)"] = _safe_float(rows[0]["LTP (₹)"] if rows else 0)
                                portfolio[pos_idx]["Qty"] = 0
                            else:
                                portfolio[pos_idx]["Qty"] = int(new_qty)
                                portfolio[pos_idx]["SL (₹)"] = new_sl
                                portfolio[pos_idx]["TGT (₹)"] = new_tgt
                                portfolio[pos_idx]["Buy Price (₹)"] = new_buy
                                portfolio[pos_idx]["Invested (₹)"] = round(new_buy * new_qty, 2)
                            st.session_state["paper_portfolio"] = portfolio
                            save_json_file(PORTFOLIO_FILE, portfolio)
                            st.success("Saved"); st.rerun()
                        if do_del and pos_idx is not None:
                            portfolio.pop(pos_idx)
                            st.session_state["paper_portfolio"] = portfolio
                            save_json_file(PORTFOLIO_FILE, portfolio)
                            st.success("Deleted"); st.rerun()

            st.markdown("---")
            st.markdown("#### Rebalance Rules")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                strategy = st.selectbox("Strategy", ["Equal Weight", "Max Position Cap", "Score / Signal Weight", "Hybrid (Equal + Cap)"])
            with c2:
                max_w = st.slider("Max weight %", 3.0, 25.0, 8.0, 0.5)
            with c3:
                drift_th = st.slider("Drift threshold %", 0.5, 10.0, 2.0, 0.5)
            with c4:
                min_tv = st.number_input("Min trade ₹", value=2000, min_value=0, step=500)

            score_map = {}
            if not df_raw.empty and "Composite Score" in df_raw.columns:
                for _, rw in df_raw.iterrows():
                    score_map[rw.get("Raw_Ticker")] = _safe_float(rw.get("Composite Score"), 50)

            n = len(rows)
            eq = 100.0 / n if n else 0
            for r in rows:
                if strategy == "Equal Weight":
                    r["Target %"] = round(eq, 2)
                elif strategy == "Max Position Cap":
                    r["Target %"] = min(r["Weight %"], max_w)
                elif strategy == "Score / Signal Weight":
                    r["_score"] = max(score_map.get(r["Raw_Ticker"], 50), 1)
                else:
                    r["Target %"] = round(min(eq, max_w), 2)

            if strategy == "Score / Signal Weight":
                ts = sum(r.get("_score", 50) for r in rows) or 1
                for r in rows:
                    r["Target %"] = round(min((r["_score"] / ts) * 100, max_w), 2)
                s = sum(r["Target %"] for r in rows) or 1
                for r in rows:
                    r["Target %"] = round(r["Target %"] / s * 100, 2)
            elif strategy == "Max Position Cap":
                s = sum(r["Target %"] for r in rows)
                residual = max(0, 100 - s)
                under = [r for r in rows if r["Weight %"] < max_w]
                if under and residual > 0:
                    add = residual / len(under)
                    for r in under:
                        r["Target %"] = round(min(r["Target %"] + add, max_w), 2)

            proposals = []
            for r in rows:
                r["Drift %"] = round(r["Weight %"] - r["Target %"], 2)
                tv = total_val * r["Target %"] / 100
                dv = tv - r["Value (₹)"]
                if abs(r["Drift %"]) < drift_th:
                    r["Action"], r["Δ Qty"] = "Hold", 0
                elif dv > min_tv and r["LTP (₹)"] > 0:
                    r["Action"] = "Buy"
                    r["Δ Qty"] = int(dv // r["LTP (₹)"])
                elif dv < -min_tv and r["LTP (₹)"] > 0:
                    r["Action"] = "Sell"
                    r["Δ Qty"] = min(int(abs(dv) // r["LTP (₹)"]), r["Qty"])
                else:
                    r["Action"], r["Δ Qty"] = "Hold", 0
                r["Δ Value (₹)"] = round(r["Δ Qty"] * r["LTP (₹)"] * (1 if r["Action"] == "Buy" else -1), 2) if r["Δ Qty"] else 0.0
                if r["Action"] != "Hold" and r["Δ Qty"] > 0:
                    proposals.append(r)

            disp = pd.DataFrame([{
                "Ticker": r["Ticker"], "Qty": r["Qty"],
                "LTP (₹)": f"₹{r['LTP (₹)']:,.2f}", "Value (₹)": f"₹{r['Value (₹)']:,.2f}",
                "Weight %": f"{r['Weight %']:.2f}%", "Target %": f"{r['Target %']:.2f}%",
                "Drift %": f"{r['Drift %']:+.2f}%", "Action": r["Action"],
                "Δ Qty": r["Δ Qty"] if r["Δ Qty"] else "—",
                "Δ Value (₹)": f"₹{r['Δ Value (₹)']:,.2f}" if r["Δ Qty"] else "—",
            } for r in rows])
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown(f"**Proposed:** {sum(1 for r in proposals if r['Action']=='Buy')} buys · {sum(1 for r in proposals if r['Action']=='Sell')} sells")

            if st.button("🔄 Apply Rebalance to Paper Portfolio", type="primary", use_container_width=True, disabled=not proposals):
                id_map = {p.get("id"): p for p in st.session_state["paper_portfolio"]}
                applied = 0
                for r in proposals:
                    pos = id_map.get(r["id"])
                    if not pos:
                        continue
                    old_q = int(pos.get("Qty", 1) or 1)
                    if r["Action"] == "Buy":
                        nq = old_q + r["Δ Qty"]
                        old_b = _safe_float(pos.get("Buy Price (₹)"), r["LTP (₹)"])
                        nb = (old_b * old_q + r["LTP (₹)"] * r["Δ Qty"]) / nq if nq else old_b
                        pos["Buy Price (₹)"] = round(nb, 2)
                        pos["Qty"] = nq
                        pos["Invested (₹)"] = round(nb * nq, 2)
                        applied += 1
                    elif r["Action"] == "Sell":
                        nq = max(0, old_q - r["Δ Qty"])
                        if nq == 0:
                            pos["Status"] = "⚪ Sold Manually"
                            pos["Exit_Date"] = str(date.today())
                            pos["Exit Price (₹)"] = r["LTP (₹)"]
                            pos["Qty"] = 0
                        else:
                            pos["Qty"] = nq
                            pos["Invested (₹)"] = round(_safe_float(pos.get("Buy Price (₹)")) * nq, 2)
                        applied += 1
                save_json_file(PORTFOLIO_FILE, st.session_state["paper_portfolio"])
                st.success(f"Applied to {applied} position(s)"); st.rerun()

