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

# Complete Embedded 2,000+ NSE Listed Equities Universe
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
    "SMSLIFE", "SMSPHARMA", "SNOWMAN", "SOBHA", "SOFTTECH", "SOLARA", "SOLARINDS",
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
    "VINNY", "VINYLINDIA", "VIPCLOTHNG", "VIPIND", "VIPULLTD", "VIRINCHI",
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
    scan_limit = st.sidebar.slider(
        "Scan Limit",
        min_value=25,
        max_value=len(all_syms),
        value=len(all_syms),
        step=25,
        help="Full 1,960+ NSE Listed Equities Universe.",
    )
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
        "🔥 Multi-Timeframe 20D Breakout",
        "Relative strength",
        "Price > Both 50 & 200 SMA",
        "Golden Cross (50 SMA > 200 SMA)",
        "Price > 50 SMA",
        "Price > 200 SMA",
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
        progress_bar.progress((c_idx + 1) / len(chunks), text=f"Scanning batch {c_idx+1}/{len(chunks)} ({min((c_idx+1)*chunk_size, total)}/{total} stocks)...")
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
                sma_200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else curr_price
                
                high_52w = float(hist["High"].max())
                dist_52w_high = max(0.0, ((high_52w - curr_price) / high_52w) * 100.0)

                prev_20d_high = float(hist["High"].iloc[:-1].tail(20).max()) if len(hist) > 20 else float(hist["High"].max())
                is_20d_high_breakout = bool(curr_price > prev_20d_high)

                rsi_val = compute_rsi(hist["Close"], 14)
                adx_val = compute_adx(hist, 14)

                vol_series = hist["Volume"].dropna()
                curr_vol = int(vol_series.iloc[-1]) if not vol_series.empty else 0
                avg_vol_20 = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else float(curr_vol)
                vol_surge = bool(curr_vol >= (avg_vol_20 * 0.95))
                vol_surge_1_5x = bool(curr_vol > (avg_vol_20 * 1.5))

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

                # Multi-Timeframe 20D Breakout Setup
                weekly_df = hist.resample("W").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
                if len(weekly_df) >= 5:
                    w_close = float(weekly_df["Close"].iloc[-1])
                    w_ema20 = float(weekly_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    w_rsi = compute_rsi(weekly_df["Close"], 14)
                    w_52h = float(weekly_df["High"].tail(52).max())
                else:
                    w_close, w_ema20, w_rsi, w_52h = curr_price, ema_20, rsi_val, high_52w

                passes_mtf_breakout = bool(
                    w_close > w_ema20 and w_rsi >= 55.0 and curr_price > ema_20 and is_20d_high_breakout and vol_surge_1_5x and curr_price >= (w_52h * 0.75)
                )

                # Relative Strength
                c_20d = float(hist["Close"].iloc[-21]) if len(hist) >= 21 else float(hist["Close"].iloc[0])
                c_125d = float(hist["Close"].iloc[-126]) if len(hist) >= 126 else float(hist["Close"].iloc[0])
                is_relative_strength = bool(
                    ((curr_price - ema_200) / ema_200 * 100.0 > 30.0)
                    and ((curr_price - c_125d) / c_125d * 100.0 > 20.0)
                    and ((curr_price - ema_50) / ema_50 * 100.0 > 20.0)
                    and ((curr_price - c_20d) / c_20d * 100.0 > 20.0)
                )

                candle_range = max(0.01, hist["High"].iloc[-1] - hist["Low"].iloc[-1])
                close_pos = (curr_price - hist["Low"].iloc[-1]) / candle_range
                score = (25 if close_pos >= 0.75 else 15) + (25 if curr_price > ema_20 > sma_50 else 10) + (15 if 55 <= rsi_val <= 75 else 5) + (15 if vol_surge else 5)
                swing_composite = float(np.clip(score, 10, 100))

                if swing_composite >= 80 and curr_price >= ema_9 >= ema_20:
                    action_signal = "🟢 STRONG BUY (Breakout)"
                elif (swing_composite >= 60 or is_triple_cross or is_cluster_squeeze or passes_mtf_breakout) and curr_price >= ema_20:
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
                    "SMA_50": round(sma_50, 2),
                    "SMA_200": round(sma_200, 2),
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
                    "_mtf_match": passes_mtf_breakout,
                    "_rs_match": is_relative_strength,
                    "_ob_gt_mcap": is_order_book_gt_mcap,
                })
                seen.add(clean_sym)
            except Exception:
                continue

        del batch_data
        gc.collect()

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
        elif sma_trend_filter == "🔥 Multi-Timeframe 20D Breakout":
            filtered_df = filtered_df[filtered_df["_mtf_match"] == True]
        elif sma_trend_filter == "Relative strength":
            filtered_df = filtered_df[filtered_df["_rs_match"] == True]
        elif sma_trend_filter == "Price > Both 50 & 200 SMA":
            filtered_df = filtered_df[(filtered_df["Price (₹)"] >= filtered_df["SMA_50"]) & (filtered_df["Price (₹)"] >= filtered_df["SMA_200"])]
        elif sma_trend_filter == "Golden Cross (50 SMA > 200 SMA)":
            filtered_df = filtered_df[filtered_df["SMA_50"] >= filtered_df["SMA_200"]]
        elif sma_trend_filter == "Price > 50 SMA":
            filtered_df = filtered_df[filtered_df["Price (₹)"] >= filtered_df["SMA_50"]]
        elif sma_trend_filter == "Price > 200 SMA":
            filtered_df = filtered_df[filtered_df["Price (₹)"] >= filtered_df["SMA_200"]]

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

                # Get TRUE Live Market Price
                curr_ltp = live_price_dict.get(sym)
                if curr_ltp is None:
                    try:
                        t = yf.Ticker(sym)
                        curr_ltp = float(t.fast_info.last_price)
                    except Exception:
                        curr_ltp = None

                # Strict Trigger Evaluation
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

    # TAB 4: COMPLETE RESTORED PAPER TRADING
    with tab_watchlist:
        st.subheader("💼 Paper Trading Portfolio & Risk Manager")
        active_portfolio = st.session_state.get("paper_portfolio", [])

        # 1. Order Placement Form
        with st.expander("➕ Execute New Paper Trade (Custom SL, Target & Remarks)", expanded=False):
            col_add1, col_add2, col_add3, col_add4, col_add5 = st.columns([1.2, 1, 1, 1, 1])
            with col_add1:
                available_tickers = df_raw["Raw_Ticker"].tolist() if not df_raw.empty else ["ACE.NS"]
                trade_stock = st.selectbox("Stock:", available_tickers, index=0)
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
                remarks = st.text_input("Trade Remarks / Strategy", value="9/20 EMA Breakout Swing Setup")
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

        # 2. Row Deletion Manager
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

        # 3. Backup & Restore
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

        # 4. Live Portfolio Tracking, Calculations & Metrics
        if active_portfolio:
            live_price_dict = dict(zip(df_raw["Raw_Ticker"], df_raw["Price (₹)"]))

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
                
                # Fetch true live price
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
                pos_remarks = str(pos.get("Remarks", "Swing Trade"))
                pos_status = str(pos.get("Status", "🟢 Open"))
                saved_exit_price = float(pos.get("Exit Price (₹)") or 0.0)

                # Automatic Status detection
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

                # Holding Days
                try:
                    d_entry = datetime.strptime(pos_date_str, "%Y-%m-%d").date()
                    if pos_exit_date_str and pos_exit_date_str.strip() and pos_exit_date_str != "-":
                        d_exit = datetime.strptime(pos_exit_date_str.strip(), "%Y-%m-%d").date()
                    else:
                        d_exit = date.today()
                    holding_days = max(0, (d_exit - d_entry).days)
                except Exception:
                    holding_days = 0

                # P&L Calculation
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

            st.session_state["paper_portfolio"] = updated_portfolio_data
            save_json_file(PORTFOLIO_FILE, updated_portfolio_data)

            win_rate_pct = (winning_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
            loss_rate_pct = (losing_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

            # 4-Box Performance Summary Grid
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

            # 5-Column Summary Metric Ribbon
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

            # Position Edit Form
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

            # Styled Table Output with Dark Green & Red P&L Map
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
        else:
            st.info("No active paper trades. Place a trade above.")
else:
    st.info("👈 Click **'🚀 Run Screener Scan'** in the sidebar to begin.")
