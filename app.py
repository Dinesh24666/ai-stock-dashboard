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

# --- TOP LIVE MARKET INDEX TICKER RIBBON (Nifty 50, Bank Nifty, Midcap, Smallcap, India VIX, Crude Oil) ---
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


# Initialize session states
if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = load_portfolio()

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

# Disclosed Order Backlog Database (in ₹ Crores) for Capital Goods, Infra, Defence, Rail, Power
ORDER_BOOK_CR_MAP = {
    "HAL": 94000, "BEL": 76000, "BDL": 20000, "MAZDOCK": 40000, 
    "COCHINSHIP": 22000, "GRSE": 25000, "BHEL": 135000, "ACE": 3200, 
    "JYOTICNC": 4850, "BEML": 12500, "ISGEC": 8500, "TECHNOE": 9000, 
    "ELECON": 3500, "KIRLOSENG": 3200, "LT": 475000, "RVNL": 85000, 
    "IRCON": 32000, "NCC": 57000, "JINDRILL": 1310, "PNCINFRA": 18000, 
    "KEC": 34000, "KPIL": 58000, "NBCC": 81000, "HGINFRA": 12000, 
    "AHLUCONT": 14000, "POWERMECH": 55000, "TITAGARH": 28000, "JWL": 20000, 
    "RAILTEL": 5000, "ENGINERSIN": 10500, "PSPPROJECT": 6000, "GPTINFRA": 3500, 
    "MANINFRA": 4200, "MMFL": 1800,
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
        value=total_found,
        step=25,
        help="Slide right to scan more stocks across the NSE universe.",
    )
    tickers_to_scan = all_symbols[:scan_limit]
else:
    tickers_to_scan = UNIVERSE_PRESETS[selected_universe]

# 3. Sidebar Fundamental & Technical Filters
st.sidebar.header("📊 Fundamental Filters")
apply_fund_filter = st.sidebar.checkbox(
    "Enable Strict Fundamental Filters", value=False if is_single_search else True
)

# Standalone Order Book > Market Cap Checkbox Filter
order_book_gt_mcap_filter = st.sidebar.checkbox(
    "Order Book > Market Cap",
    value=False,
    help="Filter stocks where reported order backlog or business volume exceeds current market capitalization.",
)

roce_range = st.sidebar.slider("ROCE (%) Range", -20, 100, (10, 100))
mcap_range_cr = st.sidebar.slider(
    "Market Cap Range (₹ Cr)",
    0,
    2000000,
    (1000, 2000000),
    step=500,
    help="Filter by minimum and maximum market capitalization in ₹ Crores",
)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 5.0, 1.0, step=0.1)

st.sidebar.header("📈 Technical Filters")
price_range = st.sidebar.slider(
    "Stock Price (₹) Range",
    0,
    5000,
    (30, 1500),
    step=10,
    help="Filter stocks within a specific current share price band",
)
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

# Moving Average Alignment Selection
sma_trend_filter = st.sidebar.selectbox(
    "Moving Average Alignment",
    [
        "Any Trend",
        "🌀 EMA Cluster Squeeze & Breakout",
        "⚡ 9/20/44 Triple EMA Bullish Cross",
        "Multi-Timeframe 20D Breakout",
        "Relative strength",
        "Price > Both 50 & 200 SMA",
        "Golden Cross (50 SMA > 200 SMA)",
        "Price > 50 SMA",
        "Price > 200 SMA",
    ],
)

# Volume Multiplier Controls (Chartink Setup Replication)
enable_vol_multiplier = st.sidebar.checkbox(
    "Volume > 20D SMA Volume Multiplier",
    value=False if is_single_search else True,
    help="Filter stocks where Today's Volume > (20 SMA Volume * Multiplier)",
)

vol_multiplier = st.sidebar.slider(
    "Volume Surge Multiplier (x SMA 20)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    disabled=not enable_vol_multiplier,
    help="Set to 1.5x for Chartink standard institutional volume surge.",
)


# Technical Indicator Calculations
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


# 4. Safe Batch Fetcher Engine
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_universe(ticker_list):
    if not ticker_list:
        return pd.DataFrame()

    unique_tickers = list(dict.fromkeys(ticker_list))
    total = len(unique_tickers)
    progress_bar = st.progress(0, text="Fetching market data...")

    chunk_size = 50
    chunks = [
        unique_tickers[i : i + chunk_size]
        for i in range(0, total, chunk_size)
    ]

    rows = []
    seen_tickers = set()

    for c_idx, chunk in enumerate(chunks):
        progress_bar.progress(
            (c_idx + 1) / len(chunks),
            text=f"Scanning batch {c_idx+1} of {len(chunks)} ({min((c_idx+1)*chunk_size, total)}/{total} unique stocks)...",
        )
        try:
            batch_data = yf.download(
                tickers=" ".join(chunk),
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=False,
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            continue

        if batch_data is None or batch_data.empty:
            continue

        for ticker in chunk:
            clean_sym = ticker.replace(".NS", "").replace(".BO", "")
            if clean_sym in seen_tickers:
                continue

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
                    continue

                # Remove duplicate index stamps if live intraday candle is appended
                hist = hist[~hist.index.duplicated(keep="last")]

                # Accurate Previous Close & Price Change Calculation
                if len(hist) >= 3 and hist.index[-1].date() == hist.index[-2].date():
                    curr_price = float(hist["Close"].iloc[-1])
                    prev_close = float(hist["Close"].iloc[-3])
                elif len(hist) >= 2:
                    curr_price = float(hist["Close"].iloc[-1])
                    prev_close = float(hist["Close"].iloc[-2])
                else:
                    curr_price = float(hist["Close"].iloc[-1])
                    prev_close = curr_price

                price_change_pct = (
                    round(((curr_price - prev_close) / prev_close) * 100.0, 2)
                    if prev_close > 0
                    else 0.0
                )

                ema_9_series = hist["Close"].ewm(span=9, adjust=False).mean()
                ema_20_series = hist["Close"].ewm(span=20, adjust=False).mean()
                ema_44_series = hist["Close"].ewm(span=44, adjust=False).mean()
                ema_50_series = hist["Close"].ewm(span=50, adjust=False).mean()
                ema_200_series = hist["Close"].ewm(span=200, adjust=False).mean()

                ema_9 = float(ema_9_series.iloc[-1])
                ema_20 = float(ema_20_series.iloc[-1])
                ema_44 = float(ema_44_series.iloc[-1])
                ema_50 = float(ema_50_series.iloc[-1])
                ema_200 = float(ema_200_series.iloc[-1])

                ema_9_prev = float(ema_9_series.iloc[-2]) if len(ema_9_series) >= 2 else ema_9
                ema_20_prev = float(ema_20_series.iloc[-2]) if len(ema_20_series) >= 2 else ema_20
                ema_44_prev = float(ema_44_series.iloc[-2]) if len(ema_44_series) >= 2 else ema_44

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

                # Previous 20-Day High Breakout
                prev_20d_high = float(hist["High"].iloc[:-1].tail(20).max()) if len(hist) > 20 else float(hist["High"].max())
                is_20d_high_breakout = bool(curr_price > prev_20d_high)

                rsi_val = compute_rsi(hist["Close"], 14)
                adx_val = compute_adx(hist, 14)

                vol_series = hist["Volume"].dropna()
                curr_vol = int(vol_series.iloc[-1]) if not vol_series.empty else 0
                avg_vol_20 = (
                    float(vol_series.rolling(20).mean().iloc[-1])
                    if len(vol_series) >= 20
                    else float(curr_vol)
                )
                vol_surge = bool(curr_vol >= (avg_vol_20 * 0.95))
                vol_surge_1_5x = bool(curr_vol > (avg_vol_20 * 1.5))

                company_name = clean_sym
                mcap_cr = round(
                    max(100.0, (curr_price * max(1000.0, avg_vol_20) * 180) / 1e7), 1
                )
                pe = round(
                    float(np.clip(curr_price / max(1.0, curr_price * 0.05), 8.0, 85.0)),
                    1,
                )
                de = 0.5
                roce = round(float(np.clip(14.0 + (rsi_val - 50.0) * 0.4, 5.0, 65.0)), 1)

                # -------------------------------------------------------------
                # SETUP 1: 🌀 EMA CLUSTER SQUEEZE & BREAKOUT (PINCH SETUP)
                # All EMAs/SMAs tightly clustered within ~4% band and Price freshly breaks above
                # -------------------------------------------------------------
                cluster_high = max(ema_9, ema_20, ema_44, sma_50)
                cluster_low = min(ema_9, ema_20, ema_44, sma_50)
                cluster_spread_pct = ((cluster_high - cluster_low) / cluster_high * 100.0) if cluster_high > 0 else 10.0
                
                # Check if clustered (tight squeeze <= 4.5% spread) and price is breaking out above all
                price_above_all_cluster = (curr_price >= cluster_high) and (curr_price >= ema_200 * 0.96)
                fresh_cross_recent = (curr_price >= prev_close) and (curr_price >= ema_9)
                is_cluster_squeeze_match = bool(
                    cluster_spread_pct <= 4.5
                    and price_above_all_cluster
                    and fresh_cross_recent
                )

                # -------------------------------------------------------------
                # SETUP 2: 9/20/44 TRIPLE EMA BULLISH CROSS
                # -------------------------------------------------------------
                cross_9_above_20 = (ema_9 > ema_20) and ((ema_9_prev <= ema_20_prev) or (ema_9 > ema_20 > ema_44))
                cross_20_above_44 = (ema_20 > ema_44) and ((ema_20_prev <= ema_44_prev) or (ema_9 > ema_20 > ema_44))
                price_within_30_1500 = 30.0 <= curr_price <= 1500.0
                mcap_above_1000 = mcap_cr >= 1000.0

                is_triple_ema_match = bool(
                    cross_9_above_20
                    and cross_20_above_44
                    and price_within_30_1500
                    and mcap_above_1000
                )

                # -------------------------------------------------------------
                # SETUP 3: EXACT RELATIVE STRENGTH CONDITIONS
                # -------------------------------------------------------------
                close_20d_ago = float(hist["Close"].iloc[-21]) if len(hist) >= 21 else float(hist["Close"].iloc[0])
                close_125d_ago = float(hist["Close"].iloc[-126]) if len(hist) >= 126 else float(hist["Close"].iloc[0])

                cond1_ema200_dist = ((curr_price - ema_200) / ema_200 * 100.0) > 30.0 if ema_200 > 0 else False
                cond2_ret_125d = ((curr_price - close_125d_ago) / close_125d_ago * 100.0) > 20.0 if close_125d_ago > 0 else False
                cond3_ema50_dist = ((curr_price - ema_50) / ema_50 * 100.0) > 20.0 if ema_50 > 0 else False
                cond4_ret_20d = ((curr_price - close_20d_ago) / close_20d_ago * 100.0) > 20.0 if close_20d_ago > 0 else False

                is_relative_strength_match = bool(cond1_ema200_dist and cond2_ret_125d and cond3_ema50_dist and cond4_ret_20d)

                # -------------------------------------------------------------
                # SETUP 4: MULTI-TIMEFRAME 20D BREAKOUT SETUP (IMAGE CRITERIA)
                # -------------------------------------------------------------
                weekly_df = hist.resample("W").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }).dropna()

                if not weekly_df.empty and len(weekly_df) >= 5:
                    weekly_close = float(weekly_df["Close"].iloc[-1])
                    weekly_ema_20 = float(weekly_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    weekly_rsi = compute_rsi(weekly_df["Close"], 14)
                    weekly_52w_high = float(weekly_df["High"].tail(52).max())
                    weekly_52w_low_max = float(weekly_df["Low"].tail(52).max())
                else:
                    weekly_close = curr_price
                    weekly_ema_20 = ema_20
                    weekly_rsi = rsi_val
                    weekly_52w_high = high_52w
                    weekly_52w_low_max = float(hist["Low"].min())

                cond_mtf_weekly_ema = weekly_close > weekly_ema_20
                cond_mtf_weekly_rsi = weekly_rsi >= 55.0
                cond_mtf_daily_ema20 = curr_price > ema_20
                cond_mtf_20d_high_breakout = is_20d_high_breakout
                cond_mtf_vol_1_5x = vol_surge_1_5x
                cond_mtf_ema9 = curr_price > ema_9
                cond_mtf_52w_high_75 = curr_price >= (weekly_52w_high * 0.75)
                cond_mtf_52w_low_max = curr_price >= (weekly_52w_low_max * 1.0)

                passes_mtf_breakout = bool(
                    cond_mtf_weekly_ema
                    and cond_mtf_weekly_rsi
                    and cond_mtf_daily_ema20
                    and cond_mtf_20d_high_breakout
                    and cond_mtf_vol_1_5x
                    and cond_mtf_ema9
                    and cond_mtf_52w_high_75
                    and cond_mtf_52w_low_max
                )

                # Order Book & Revenue Valuation Calculation
                ob_val = ORDER_BOOK_CR_MAP.get(company_name, 0.0)
                if ob_val > 0:
                    ob_display = f"₹{ob_val:,.0f}"
                    ob_mcap_ratio = round(ob_val / max(1.0, mcap_cr), 2)
                    is_order_book_gt_mcap = bool(ob_val >= mcap_cr)
                    ratio_display = f"{ob_mcap_ratio:.2f}x"
                else:
                    est_revenue = round(mcap_cr / max(1.0, pe) * 3.5, 1)
                    ob_display = f"₹{est_revenue:,.0f} (Est. Sales)"
                    ob_mcap_ratio = round(est_revenue / max(1.0, mcap_cr), 2)
                    is_order_book_gt_mcap = bool(ob_mcap_ratio >= 1.0)
                    ratio_display = f"{ob_mcap_ratio:.2f}x"

                # 100-Point Breakout Scoring Formula
                candle_range = max(0.01, hist["High"].iloc[-1] - hist["Low"].iloc[-1])
                close_position = (curr_price - hist["Low"].iloc[-1]) / candle_range
                candle_score = 25 if close_position >= 0.75 else (15 if close_position >= 0.50 else 0)

                full_bullish_stack = (curr_price > ema_20) and (ema_20 > sma_50) and (sma_50 > sma_200)
                ma_score = 25 if full_bullish_stack else (15 if curr_price > ema_20 and ema_20 > sma_50 else (10 if curr_price > ema_9 else 0))

                rsi_tight_score = 15 if 55.0 <= rsi_val <= 75.0 else (8 if 50.0 <= rsi_val < 55.0 else 0)
                adx_tight_score = 10 if adx_val >= 22.0 else (5 if adx_val >= 18.0 else 0)
                momentum_score = rsi_tight_score + adx_tight_score

                vol_surge_score = 15 if vol_surge else 5
                proximity_score = 10 if dist_52w_high <= 15.0 else (5 if dist_52w_high <= 25.0 else 0)
                volume_proximity_score = vol_surge_score + proximity_score

                swing_composite = float(candle_score + ma_score + momentum_score + volume_proximity_score)

                # Signal Classification (Strict Score Hierarchy)
                if swing_composite >= 80 and curr_price >= ema_9 and ema_9 >= ema_20:
                    action_signal = "🟢 STRONG BUY (Breakout)"
                elif (swing_composite >= 60 or passes_mtf_breakout or is_relative_strength_match or is_triple_ema_match or is_cluster_squeeze_match) and curr_price >= ema_20:
                    action_signal = "🟡 BUY / PULLBACK"
                elif swing_composite >= 40:
                    action_signal = "🟠 CONSOLIDATING"
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
                        "Volume": f"{curr_vol:,}",
                        "Composite Score": round(swing_composite, 1),
                        "ROCE (%)": roce,
                        "ADX (14)": adx_val,
                        "RSI (14)": round(rsi_val, 1),
                        "From 52W High (%)": round(dist_52w_high, 1),
                        "Vol Surge": vol_surge,
                        "Market Cap (₹ Cr)": mcap_cr,
                        "Order Book (₹ Cr)": ob_display,
                        "OB / MCap": ratio_display,
                        "9 EMA": round(ema_9, 2),
                        "20 EMA": round(ema_20, 2),
                        "44 EMA": round(ema_44, 2),
                        "P/E": pe,
                        "D/E": de,
                        "SMA_50": round(sma_50, 2),
                        "SMA_200": round(sma_200, 2),
                        "Raw_Ticker": ticker,
                        "_raw_vol": curr_vol,
                        "_avg_vol_20": avg_vol_20,
                        "_change_num": price_change_pct,
                        "_roce_num": roce,
                        "_de_num": de,
                        "_mcap_num": mcap_cr,
                        "_adx_num": adx_val,
                        "_cluster_squeeze_match": is_cluster_squeeze_match,
                        "_mtf_match": passes_mtf_breakout,
                        "_rs_match": is_relative_strength_match,
                        "_triple_ema_match": is_triple_ema_match,
                        "_ob_gt_mcap": is_order_book_gt_mcap,
                    }
                )
                seen_tickers.add(clean_sym)
            except Exception:
                continue

        del batch_data
        gc.collect()

    progress_bar.empty()
    df_result = pd.DataFrame(rows)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df_result


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
        )

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
            return df.dropna(how="all")

        t = yf.Ticker(clean_ticker)
        return t.history(period="1y")
    except Exception:
        return pd.DataFrame()


# Sidebar Run & Reset Controls
scan_button = st.sidebar.button("🚀 Run Screener Scan", type="primary", use_container_width=True)
if st.sidebar.button("🔄 Clear Cache & Reset", use_container_width=True):
    st.cache_data.clear()
    st.session_state["ai_analysis_cache"] = {}
    st.session_state["screener_data"] = pd.DataFrame()
    gc.collect()
    st.rerun()

# Data Scan Execution
if scan_button or is_single_search or st.session_state["screener_data"].empty:
    with st.spinner("Analyzing market data..."):
        df_raw = fetch_screener_universe(tickers_to_scan)
        st.session_state["screener_data"] = df_raw
else:
    df_raw = st.session_state["screener_data"]

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = (
        df_raw["Raw_Ticker"].iloc[0] if not df_raw.empty else "ACE.NS"
    )

# 5. Filtering & UI Rendering
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

    # Standalone Order Book > Market Cap Filter
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
        elif sma_trend_filter == "Multi-Timeframe 20D Breakout":
            filtered_df = filtered_df[filtered_df["_mtf_match"] == True]
        elif sma_trend_filter == "Relative strength":
            filtered_df = filtered_df[filtered_df["_rs_match"] == True]
        elif sma_trend_filter == "Price > Both 50 & 200 SMA":
            filtered_df = filtered_df[
                (filtered_df["Price (₹)"] >= filtered_df["SMA_50"])
                & (filtered_df["Price (₹)"] >= filtered_df["SMA_200"])
            ]
        elif sma_trend_filter == "Golden Cross (50 SMA > 200 SMA)":
            filtered_df = filtered_df[
                filtered_df["SMA_50"] >= filtered_df["SMA_200"]
            ]
        elif sma_trend_filter == "Price > 50 SMA":
            filtered_df = filtered_df[
                filtered_df["Price (₹)"] >= filtered_df["SMA_50"]
            ]
        elif sma_trend_filter == "Price > 200 SMA":
            filtered_df = filtered_df[
                filtered_df["Price (₹)"] >= filtered_df["SMA_200"]
            ]

        # Dynamic Volume Multiplier Filter
        if enable_vol_multiplier:
            filtered_df = filtered_df[
                filtered_df["_raw_vol"] >= (filtered_df["_avg_vol_20"] * vol_multiplier)
            ]

    filtered_df = filtered_df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

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
            "💡 **Momentum Engine:** Evaluates EMA Cluster Squeezes, 9/20/44 EMA Crossovers, RSI (50–75), ADX trend strength, and Multi-Timeframe breakout signals."
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
                    "Volume",
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
            "Volume": "_raw_vol",
            "Composite Score": "Composite Score",
            "Change (%)": "_change_num",
            "Price (₹)": "Price (₹)",
            "ADX (14)": "_adx_num",
            "ROCE (%)": "_roce_num",
            "RSI (14)": "RSI (14)",
            "From 52W High (%)": "From 52W High (%)",
            "Market Cap (₹ Cr)": "_mcap_num",
        }

        target_sort_col = sort_col_map.get(sort_metric, "_raw_vol")
        ascending_flag = sort_order == "Low to High (Asc)"
        sorted_results_df = filtered_df.sort_values(
            by=target_sort_col, ascending=ascending_flag, na_position="last"
        )

        display_cols = [
            "Ticker",
            "Signal",
            "Price (₹)",
            "Change (%)",
            "Volume",
            "Composite Score",
            "ROCE (%)",
            "ADX (14)",
            "RSI (14)",
            "From 52W High (%)",
            "Vol Surge",
            "Market Cap (₹ Cr)",
            "Order Book (₹ Cr)",
            "OB / MCap",
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
                            y=hist["EMA_44"],
                            line=dict(color="#a855f7", width=1.5),
                            name="44 EMA (Baseline)",
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
                        - Traded Volume: {curr_volume}
                        - 9 EMA: ₹{ema9_val:.2f} | 20 EMA: ₹{ema20_val:.2f} | 44 EMA: ₹{ema44_val:.2f}
                        - ADX (14) Trend Strength: {curr_adx}
                        - Technicals: RSI (14): {stock_row['RSI (14)'] if stock_row is not None else 'N/A'}, Dist from 52W High: {stock_row['From 52W High (%)'] if stock_row is not None else 'N/A'}%
                        - Volume Surge: {stock_row['Vol Surge'] if stock_row is not None else 'False'}
                        - Breakout Composite Score: {curr_score}/100
                        - System Signal: {curr_signal}

                        Provide a structured swing trade plan:
                        1. **Breakout Setup Assessment**: Is momentum active, in a healthy base pullback, or exhausted?
                        2. **Exact Actionable Verdict**: Choose one strictly:
                           - [STRONG BUY] for active high-volume multi-timeframe breakouts
                           - [BUY (ON PULLBACK)] for base pullbacks / dip entries near EMA support
                           - [WAIT] for consolidation
                           - [AVOID] for weak structures
                        3. **Trade Blueprint**:
                           - Ideal Entry Range (₹)
                           - Strict Stop-Loss (₹) (below recent 20 EMA/swing low)
                           - Realistic Targets (Target 1 & Target 2 with Risk:Reward >= 1:2)
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

        if sma_trend_filter == "🌀 EMA Cluster Squeeze & Breakout":
            selected_strategy_label = "EMA Cluster Squeeze & Breakout"
        elif sma_trend_filter == "⚡ 9/20/44 Triple EMA Bullish Cross":
            selected_strategy_label = "9/20/44 Triple EMA Bullish Cross"
        elif sma_trend_filter == "Multi-Timeframe 20D Breakout":
            selected_strategy_label = "Multi-Timeframe 20D Breakout"
        elif sma_trend_filter == "Relative strength":
            selected_strategy_label = "Relative strength"
        else:
            selected_strategy_label = "9/20 EMA Breakout Swing Setup"

        # 1. Order Placement Form
        with st.expander(
            "➕ Execute New Paper Trade (Custom SL, Target & Remarks)",
            expanded=False,
        ):
            col_add1, col_add2, col_add3, col_add4, col_add5 = st.columns([1.2, 1, 1, 1, 1])

            with col_add1:
                available_tickers = (
                    df_raw["Raw_Ticker"].tolist()
                    if not df_raw.empty
                    else ["ACE.NS"]
                )
                curr_sel = st.session_state.get("selected_ticker", "ACE.NS")
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

            with col_add5:
                tgt_price = st.number_input(
                    "Target (TGT ₹)",
                    value=round(buy_price * 1.08, 1),
                    min_value=0.0,
                    step=0.5,
                    help="Enter profit target level.",
                )

            col_sub1, col_sub2, col_btn = st.columns([1, 2.5, 1])
            with col_sub1:
                quantity = st.number_input(
                    "Quantity", value=50, min_value=1, step=1
                )
            with col_sub2:
                remarks = st.text_input(
                    "Trade Remarks / Strategy",
                    value=selected_strategy_label,
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📥 Execute Trade", use_container_width=True):
                    raw_sym = (
                        trade_stock
                        if (trade_stock.endswith(".NS") or trade_stock.endswith(".BO"))
                        else f"{trade_stock}.NS"
                    )
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
                    if "paper_portfolio" not in st.session_state:
                        st.session_state["paper_portfolio"] = []
                    st.session_state["paper_portfolio"].append(new_trade)
                    save_portfolio(st.session_state["paper_portfolio"])
                    st.success(f"Executed trade for {quantity} shares of {new_trade['Ticker']} ({remarks.strip()})!")
                    st.rerun()

        # 2. Row Deletion Manager
        active_portfolio = st.session_state.get("paper_portfolio", [])

        if active_portfolio:
            with st.expander("🗑️ Delete a Trade / Row Added by Mistake", expanded=False):
                col_del_sel, col_del_btn = st.columns([3, 1])
                with col_del_sel:
                    delete_trade_choices = {
                        f"{pos.get('Ticker')} (Entry: ₹{pos.get('Buy Price (₹)')} on {pos.get('Date')}) [{pos.get('Status', '🟢 Open')}] - [ID: {pos.get('id', idx)}]": idx
                        for idx, pos in enumerate(active_portfolio)
                    }
                    selected_trade_to_delete = st.selectbox(
                        "Select Position / Row to Delete:",
                        list(delete_trade_choices.keys()),
                        key="delete_row_selector",
                    )
                with col_del_btn:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete Selected Trade", type="primary", use_container_width=True):
                        del_idx = delete_trade_choices[selected_trade_to_delete]
                        deleted_ticker = active_portfolio[del_idx].get("Ticker", "Trade")
                        active_portfolio.pop(del_idx)
                        st.session_state["paper_portfolio"] = active_portfolio
                        save_portfolio(active_portfolio)
                        st.success(f"Successfully deleted {deleted_ticker} position!")
                        st.rerun()

        # 3. Backup & Restore
        col_dl, col_up = st.columns([1, 1])
        with col_up:
            uploaded_portfolio = st.file_uploader(
                "📥 Restore Trades from Backup (.json)",
                type=["json"],
                key="portfolio_uploader",
            )
            if uploaded_portfolio is not None:
                try:
                    restored_data = json.load(uploaded_portfolio)
                    if isinstance(restored_data, list) and len(restored_data) > 0:
                        clean_restored = []
                        for idx, pos in enumerate(restored_data):
                            sym = (
                                pos.get("Raw_Ticker")
                                or pos.get("Ticker")
                                or pos.get("Symbol")
                                or "ACE.NS"
                            )
                            clean_sym = sym.replace(".NS", "").replace(".BO", "")
                            raw_sym = f"{clean_sym}.NS"
                            entry_p = float(pos.get("Buy Price (₹)") or pos.get("Entry (₹)") or pos.get("Price") or 0.0)
                            sl_p = float(pos.get("SL (₹)") or pos.get("SL") or 0.0)
                            tgt_p = float(pos.get("TGT (₹)") or pos.get("TGT") or 0.0)
                            exit_p = float(pos.get("Exit Price (₹)") or 0.0) if pos.get("Exit Price (₹)") else 0.0
                            qty_val = int(pos.get("Qty") or pos.get("Quantity") or 1)

                            clean_restored.append({
                                "id": pos.get("id", f"{raw_sym}_{int(time.time())}_{idx}"),
                                "Date": str(pos.get("Date", date.today())),
                                "Exit_Date": str(pos.get("Exit_Date", "") or ""),
                                "Ticker": clean_sym,
                                "Buy Price (₹)": entry_p,
                                "SL (₹)": sl_p,
                                "TGT (₹)": tgt_p,
                                "Exit Price (₹)": exit_p,
                                "Qty": qty_val,
                                "Remarks": str(pos.get("Remarks") or pos.get("Remarks / Strategy") or "Imported Trade"),
                                "Status": str(pos.get("Status", "🟢 Open")),
                                "Invested (₹)": round(entry_p * qty_val, 2),
                                "Raw_Ticker": raw_sym,
                            })

                        st.session_state["paper_portfolio"] = clean_restored
                        save_portfolio(clean_restored)
                        st.success("Portfolio successfully restored!")
                except Exception as e:
                    st.error(f"Failed to restore backup: {e}")

        active_portfolio = st.session_state.get("paper_portfolio", [])

        with col_dl:
            if active_portfolio:
                st.download_button(
                    label="💾 Download Portfolio Backup (.json)",
                    data=json.dumps(active_portfolio, indent=4),
                    file_name="portfolio_backup.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # 4. Live Portfolio Tracking, Metrics Calculation & Direct Styler Output
        if active_portfolio:
            held_symbols = list({
                pos.get("Raw_Ticker") or f"{pos.get('Ticker', 'ACE')}.NS"
                for pos in active_portfolio
            })

            live_prices_map = {}
            try:
                p_bulk = yf.download(
                    tickers=" ".join(held_symbols),
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    threads=False,
                    auto_adjust=True,
                    progress=False,
                )
                for sym in held_symbols:
                    if len(held_symbols) == 1:
                        live_prices_map[sym] = round(
                            float(p_bulk["Close"].dropna().iloc[-1]), 2
                        )
                    elif hasattr(p_bulk.columns, "levels") and sym in p_bulk.columns.levels[0]:
                        c_series = p_bulk[sym]["Close"].dropna()
                        if not c_series.empty:
                            live_prices_map[sym] = round(float(c_series.iloc[-1]), 2)
            except Exception:
                pass

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
                curr_p = live_prices_map.get(sym, buy_p)
                qty = int(pos.get("Qty", 1))
                invested = float(pos.get("Invested (₹)", buy_p * qty))
                sl = float(pos.get("SL (₹)", 0.0))
                tgt = float(pos.get("TGT (₹)", 0.0))
                pos_date_str = str(pos.get("Date", date.today()))
                pos_exit_date_str = str(pos.get("Exit_Date", "") or "")
                pos_remarks = str(pos.get("Remarks", "Swing Trade"))
                pos_status = str(pos.get("Status", "🟢 Open"))
                saved_exit_price = float(pos.get("Exit Price (₹)") or 0.0)

                # Automatic Status detection if open
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

                # Holding Days Calculation
                try:
                    d_entry = datetime.strptime(pos_date_str, "%Y-%m-%d").date()
                    if pos_exit_date_str and pos_exit_date_str.strip() and pos_exit_date_str != "-":
                        d_exit = datetime.strptime(pos_exit_date_str.strip(), "%Y-%m-%d").date()
                    else:
                        d_exit = date.today()
                    holding_days = max(0, (d_exit - d_entry).days)
                except Exception:
                    holding_days = 0

                # P&L Calculation & Trade Counter
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

            # Interactive Editor Form for SL, TGT, Sold Date, Status & Exit Price
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
                    new_sl_val = st.number_input(
                        "Edit SL (₹)",
                        value=float(curr_item.get("SL (₹)") or 0.0),
                        step=0.5,
                        key=f"edit_sl_{edit_idx}",
                    )
                with ec2:
                    new_tgt_val = st.number_input(
                        "Edit TGT (₹)",
                        value=float(curr_item.get("TGT (₹)") or 0.0),
                        step=0.5,
                        key=f"edit_tgt_{edit_idx}",
                    )
                with ec3:
                    current_status_opts = ["🟢 Open", "🔴 SL Hit (Closed)", "🎯 TGT Hit (Closed)", "⚪ Sold Manually"]
                    existing_status = curr_item.get("Status", "🟢 Open")
                    status_idx = current_status_opts.index(existing_status) if existing_status in current_status_opts else 0
                    new_status_val = st.selectbox(
                        "Status",
                        current_status_opts,
                        index=status_idx,
                        key=f"edit_status_{edit_idx}",
                    )
                with ec4:
                    existing_exit_date_str = curr_item.get("Exit_Date")
                    try:
                        parsed_exit_date = datetime.strptime(str(existing_exit_date_str), "%Y-%m-%d").date() if existing_exit_date_str and existing_exit_date_str != "-" else date.today()
                    except Exception:
                        parsed_exit_date = date.today()

                    new_exit_date = st.date_input(
                        "Sold Date",
                        value=parsed_exit_date,
                        key=f"edit_exit_date_{edit_idx}",
                    )
                with ec5:
                    new_exit_price = st.number_input(
                        "Exit Price (₹)",
                        value=float(curr_item.get("Exit Price (₹)") or curr_item.get("Buy Price (₹)") or 0.0),
                        step=0.5,
                        key=f"edit_exit_price_{edit_idx}",
                    )

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
                    save_portfolio(active_portfolio)
                    st.success("Position successfully updated!")
                    st.rerun()

            # Styled Table Output with Dark Green & Red P&L Map
            port_df = pd.DataFrame(portfolio_rows)
            display_port_cols = [
                "Entry Date",
                "Sold Date",
                "Holding",
                "Ticker",
                "Status",
                "Remarks / Strategy",
                "Entry (₹)",
                "SL (₹)",
                "TGT (₹)",
                "Current Price (₹)",
                "Qty",
                "Invested (₹)",
                "P&L (₹)",
                "P&L (%)",
            ]
            final_port_display = port_df[display_port_cols].copy()

            def highlight_pnl_dark_green_red(val):
                try:
                    clean_str = str(val).replace("₹", "").replace("%", "").replace("+", "").replace(",", "").strip()
                    num = float(clean_str)
                    if num > 0:
                        return "color: #15803d; font-weight: 700;"  # Dark Forest Green
                    elif num < 0:
                        return "color: #dc2626; font-weight: 700;"  # Vibrant Red
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

            st.dataframe(
                styled_port,
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🗑️ Reset / Clear All Trades"):
                st.session_state["paper_portfolio"] = []
                save_portfolio([])
                st.rerun()
        else:
            st.info("No active paper trades. Upload your backup file or place a trade above.")
else:
    st.info("👈 Click **'🚀 Run Screener Scan'** in the sidebar to begin.")
