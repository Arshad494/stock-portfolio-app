import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import sqlite3
import datetime
import requests
import json
import io
import time
from io import BytesIO

# Try to import AI SDKs
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# --- 1. CONFIGURATION & SESSION STATE ---

st.set_page_config(
    page_title="Arshad Stock Portfolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
def init_session_state():
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame()
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = pd.DataFrame()
    # Ensure database table exists
    init_db()

# --- 2. DATABASE & UTILS ---

DB_FILE = "portfolio.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS holdings
                 (ticker TEXT PRIMARY KEY, quantity REAL, avg_price REAL, sector TEXT, last_updated TIMESTAMP)''')
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect(DB_FILE)

def format_inr(amount):
    """Format currency in Indian Rupee format."""
    return f"₹{amount:,.2f}"

# --- 3. DATA LOADING & PROCESSING ---

DEFAULT_TICKER_MAPPING = {
    "ETERNAL": "TATAELXSI.NS", 
    "CUMMINS": "CUMMINSIND.NS",
    "CUMMINS INDIA": "CUMMINSIND.NS",
    "CUMMINS INDIA LTD": "CUMMINSIND.NS",
    "MAHINDRA AND MAHINDRA": "M&M.NS",
    "M&M": "M&M.NS",
    "CROMPTON GREAVES": "CROMPTON.NS",
    "CROMPTON": "CROMPTON.NS",
    "KAYNES TECH": "KAYNES.NS",
    "KAYNES": "KAYNES.NS",
    "BILLIONBRAINS GARAGE VN L": "GROWW.NS", 
    "GROWW": "GROOW.NS", 
    "SAMVARDHANA MOTHERSON": "MOTHERSON.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "METALIETF": "METALIETF.NS",
    "UNO MINDA": "UNOMINDA.NS",
    "SONA BLW PRECISION": "SONACOMS.NS",
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "INFOSYS": "INFY.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "ITC HOTELS LIMITED": "ITCHOTELS.NS", 
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "TITAN": "TITAN.NS"
}

def normalize_ticker(name_or_ticker):
    """Robust mapping of company names to Yahoo Finance Tickers."""
    if pd.isna(name_or_ticker) or str(name_or_ticker).strip() == "":
        return None
        
    original = str(name_or_ticker).strip()
    upper_name = original.upper()
    
    if upper_name in DEFAULT_TICKER_MAPPING: return DEFAULT_TICKER_MAPPING[upper_name]
    if upper_name.endswith(".NS") or upper_name.endswith(".BO"): return upper_name
        
    clean_name = upper_name
    suffixes = [" LIMITED", " LTD", " INDIA", " IND", " PRIVATE", " PVT", " CO", " COMPANY"]
    for s in suffixes:
        if clean_name.endswith(s): clean_name = clean_name[:-len(s)].strip()
            
    clean_name = clean_name.replace(".", "").replace(",", "")
    if " " not in clean_name and len(clean_name) < 15: return f"{clean_name}.NS"
    return f"{clean_name.replace(' ', '')}.NS"

@st.cache_data(ttl=300)
def get_realtime_price(ticker):
    """Fetch current price with caching."""
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
        # Fallback to daily
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    """Fetch basic info like sector."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}

def load_and_process_file(uploaded_file):
    """Process uploaded file with broker auto-detection."""
    try:
        # Load logic based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format."

        # Auto-detect Broker Columns
        col_map = {
            "Ticker": ["Ticker", "Symbol", "Stock Symbol", "Scrip", "Instrument"],
            "Stock Name": ["Stock Name", "Company", "Company Name", "Name", "Instrument Name", "Scrip Name"],
            "Quantity": ["Quantity", "Qty", "Units", "Shares", "Balance", "Qty."],
            "AvgPrice": ["Average buy price", "Avg Price", "Buy Price", "Cost Basis", "Avg. Cost", "Avg. Price", "Avg.", "Average Price"]
        }
        
        found = {}
        for target, alts in col_map.items():
            for alt in alts:
                match = next((c for c in df.columns if c.lower() == alt.lower()), None)
                if match: 
                    found[target] = match; break
        
        final_df = pd.DataFrame()
        
        # Determine Ticker Source
        if "Ticker" in found: 
            final_df['Ticker'] = df[found["Ticker"]]
        elif "Stock Name" in found: 
            final_df['Ticker'] = df[found["Stock Name"]]
        else: 
            return None, "File missing Ticker/Name column."
            
        if "Quantity" in found: 
            final_df['Quantity'] = pd.to_numeric(df[found["Quantity"]], errors='coerce').fillna(0)
        else: 
            return None, "Quantity missing."
            
        if "AvgPrice" in found: 
            final_df['AvgPrice'] = pd.to_numeric(df[found["AvgPrice"]], errors='coerce').fillna(0.0)
        else: 
            return None, "Avg Price missing."

        # Normalize Tickers
        final_df['Ticker'] = final_df['Ticker'].apply(normalize_ticker)
        final_df = final_df.dropna(subset=['Ticker'])
        final_df = final_df[final_df['Ticker'] != '']
        
        # Save to DB
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM holdings") # Overwrite mode
        for _, row in final_df.iterrows():
            c.execute("INSERT OR REPLACE INTO holdings (ticker, quantity, avg_price, last_updated) VALUES (?, ?, ?, ?)",
                      (row['Ticker'], row['Quantity'], row['AvgPrice'], datetime.datetime.now()))
        conn.commit()
        conn.close()
        
        return final_df, None
        
    except Exception as e:
        return None, f"Error processing file: {e}"

def get_portfolio_metrics():
    """Calculate all portfolio metrics from DB."""
    conn = get_db_connection()
    df = pd.read_sql("SELECT ticker, quantity, avg_price FROM holdings", conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0
        
    metrics = []
    
    # Progress for large portfolios
    progress_text = "Parsing Portfolio..."
    my_bar = st.progress(0, text=progress_text)
    
    total_holdings = len(df)
    
    for i, (index, row) in enumerate(df.iterrows()):
        ticker = row['ticker']
        qty = row['quantity']
        avg = row['avg_price']
        
        curr = get_realtime_price(ticker)
        # If caching misses and API fails, assume unchanged (safeguard)
        if curr == 0: curr = avg 
        
        val = qty * curr
        inv = qty * avg
        pnl = val - inv
        pnl_pct = (pnl/inv)*100 if inv > 0 else 0
        
        info = get_stock_info(ticker)
        sector = info.get('sector', 'Unknown')
        name = info.get('shortName', ticker)
        
        metrics.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Quantity": qty,
            "Avg Price": avg,
            "Current Price": curr,
            "Invested": inv,
            "Value": val,
            "P&L": pnl,
            "P&L %": pnl_pct
        })
        
        my_bar.progress((i + 1) / total_holdings, text=f"Fetching data for {ticker}...")
        
    my_bar.empty()
    
    metrics_df = pd.DataFrame(metrics)
    total_inv = metrics_df['Invested'].sum()
    total_val = metrics_df['Value'].sum()
    
    return metrics_df, df, total_inv, total_val

def export_to_excel(metrics_df):
    """Export portfolio to Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        metrics_df.to_excel(writer, sheet_name='Holdings', index=False)
        # Add summary sheet
        summary = pd.DataFrame({
            'Metric': ['Total Invested', 'Current Value', 'P&L'],
            'Value': [metrics_df['Invested'].sum(), metrics_df['Value'].sum(), metrics_df['P&L'].sum()]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    return output.getvalue()


# --- 4. AI INTEGRATION ---

def query_ai(provider, api_key, model, context, question):
    system_prompt = "You are an expert financial assistant for Arshad's Portfolio. Analyze the provided portfolio data. Be concise, actionable, and data-driven."
    context_str = json.dumps(context, indent=2)
    
    if provider == "OpenAI" and HAS_OPENAI:
        try:
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Context: {context_str}\n\nQ: {question}"}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"
        
    elif provider == "Google Gemini" and HAS_GEMINI:
        try:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            resp = m.generate_content(f"{system_prompt}\nContext: {context_str}\nQ: {question}")
            return resp.text
        except Exception as e: return f"Error: {e}"
        
    return "Provider not available."


# --- 5. MAIN UI ---

init_session_state()

st.sidebar.title("💼 Arshad's Portfolio")
nav = st.sidebar.radio("Navigation", ["Dashboard", "Portfolio Manager", "Analytics & Reports", "AI Assistant", "Settings"], index=0)

# --- DASHBOARD ---
if nav == "Dashboard":
    st.title("📊 Arshad Stock Portfolio")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    metrics_df, raw_df, total_inv, total_val = get_portfolio_metrics()
    
    if not metrics_df.empty:
        # Core Metrics
        total_pnl = total_val - total_inv
        total_pnl_pct = (total_pnl / total_inv) * 100 if total_inv > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Investment", format_inr(total_inv))
        c2.metric("Current Value", format_inr(total_val), delta=f"{total_pnl_pct:.2f}%")
        c3.metric("Total P&L", format_inr(total_pnl), delta=f"{total_pnl_pct:.2f}%")
        c4.metric("Holdings", f"{len(metrics_df)}")
        
        st.markdown("---")
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.subheader("Sector Allocation")
            fig_sec = px.pie(metrics_df, values='Value', names='Sector', title='Sector-wise Distribution', hole=0.4)
            st.plotly_chart(fig_sec, use_container_width=True)
            
        with col_viz2:
            st.subheader("Top 10 Holdings")
            top10 = metrics_df.nlargest(10, 'Value')
            fig_top = px.bar(top10, x='Name', y='Value', color='P&L %', 
                             title='Top 10 by Value',
                             color_continuous_scale=['red', 'yellow', 'green'])
            st.plotly_chart(fig_top, use_container_width=True)
            
        st.subheader("Portfolio Composition")
        fig_tree = px.treemap(metrics_df, path=['Sector', 'Name'], values='Value', 
                              color='P&L %', color_continuous_scale='RdYlGn',
                              title='Portfolio Heatmap')
        st.plotly_chart(fig_tree, use_container_width=True)
            
    else:
        st.info("👋 Welcome Arshad! Go to **Portfolio Manager** to upload your holdings.")
        if st.button("Load Sample Data"):
            sample_data = pd.DataFrame([
                {"Ticker": "TATAELXSI.NS", "Quantity": 50, "AvgPrice": 7200},
                {"Ticker": "RELIANCE.NS", "Quantity": 100, "AvgPrice": 2400},
                {"Ticker": "INFY.NS", "Quantity": 200, "AvgPrice": 1450},
                {"Ticker": "HDFCBANK.NS", "Quantity": 150, "AvgPrice": 1550}
            ])
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM holdings")
            for _, row in sample_data.iterrows():
                c.execute("INSERT INTO holdings (ticker, quantity, avg_price, last_updated) VALUES (?, ?, ?, ?)",
                          (row['Ticker'], row['Quantity'], row['AvgPrice'], datetime.datetime.now()))
            conn.commit()
            conn.close()
            st.success("Sample Data Loaded! Please Refresh.")

# --- PORTFOLIO MANAGER ---
elif nav == "Portfolio Manager":
    st.title("🗃️ Portfolio Manager")
    
    tab1, tab2 = st.tabs(["Import File", "Manual Edit"])
    
    with tab1:
        st.subheader("Upload Broker Export")
        st.info("Supported: Groww (CSV), Zerodha (Excel), Upstox (CSV)")
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            with st.spinner("Processing file..."):
                df, err = load_and_process_file(uploaded_file)
                if err:
                    st.error(err)
                else:
                    st.success(f"✅ Successfully imported {len(df)} holdings!")
                    st.dataframe(df.head())
                    
    with tab2:
        st.subheader("Edit Holdings")
        conn = get_db_connection()
        curr_df = pd.read_sql("SELECT ticker, quantity, avg_price FROM holdings", conn)
        conn.close()
        
        edited_df = st.data_editor(curr_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("Save Changes"):
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM holdings")
            for _, row in edited_df.iterrows():
                c.execute("INSERT OR REPLACE INTO holdings (ticker, quantity, avg_price, last_updated) VALUES (?, ?, ?, ?)",
                          (row['ticker'], row['quantity'], row['avg_price'], datetime.datetime.now()))
            conn.commit()
            conn.close()
            st.success("Changes Saved!")

# --- ANALYTICS & REPORTS ---
elif nav == "Analytics & Reports":
    st.title("📈 Analytics & Tax")
    
    metrics_df, _, _, _ = get_portfolio_metrics()
    
    if metrics_df.empty:
        st.warning("No data available.")
        st.stop()
        
    st.subheader("Performance Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        best = metrics_df.loc[metrics_df['P&L %'].idxmax()]
        st.metric("🌟 Best Performer", best['Name'], f"{best['P&L %']:.2f}%")
    with col2:
        worst = metrics_df.loc[metrics_df['P&L %'].idxmin()]
        st.metric("📉 Worst Performer", worst['Name'], f"{worst['P&L %']:.2f}%")
        
    st.markdown("---")
    
    with st.expander("💰 Tax Calculator (Estimated)", expanded=True):
        st.info("Estimates based on Unrealized Gains. Assumes STCG=15%, LTCG=12.5% (>1L).")
        total_gain = metrics_df['P&L'].sum()
        
        if total_gain > 0:
            stcg_gain = total_gain * 0.5
            ltcg_gain = total_gain * 0.5
            stcg_tax = stcg_gain * 0.15
            ltcg_tax = max(0, ltcg_gain - 100000) * 0.125
            
            t1, t2, t3 = st.columns(3)
            t1.metric("Est. STCG Tax", format_inr(stcg_tax))
            t2.metric("Est. LTCG Tax", format_inr(ltcg_tax))
            t3.metric("Total Tax Liab.", format_inr(stcg_tax + ltcg_tax))
        else:
            st.success("No Tax Liability (Overall Loss)")

    st.markdown("---")
    st.subheader("Export Report")
    
    excel_data = export_to_excel(metrics_df)
    st.download_button(
        label="📥 Download Portfolio Report (Excel)",
        data=excel_data,
        file_name=f"arshad_portfolio_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- AI ASSISTANT ---
elif nav == "AI Assistant":
    st.title("🧠 AI Insights")
    metrics_df, _, _, _ = get_portfolio_metrics()
    
    c1, c2 = st.columns(2)
    provider = c1.selectbox("Provider", ["OpenAI", "Google Gemini"])
    api_key_input = c2.text_input("API Key", type="password")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello Arshad! Ask me anything about your portfolio."}]
        
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        context = {
            "summary": metrics_df[['Name', 'Sector', 'Value', 'P&L %']].to_dict(orient='records') if not metrics_df.empty else "No Data"
        }
        
        with st.spinner("AI Thinking..."):
            ans = query_ai(provider, api_key_input, "gpt-3.5-turbo" if provider=="OpenAI" else "gemini-2.0-flash", context, prompt)
            
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").write(ans)

# --- SETTINGS ---
elif nav == "Settings":
    st.title("⚙️ Settings")
    st.subheader("Preferences")
    currency = st.selectbox("Currency", ["INR (₹)", "USD ($)"])
    tax_stcg = st.slider("STCG Rate (%)", 0, 30, 15)
    tax_ltcg = st.slider("LTCG Rate (%)", 0, 30, 12)
    
    st.subheader("Data Management")
    if st.button("Reset Application"):
        conn = get_db_connection()
        conn.execute("DELETE FROM holdings")
        conn.commit()
        conn.close()
        st.cache_data.clear()
        st.warning("All data cleared.")
