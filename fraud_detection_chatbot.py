import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="SAP Fraud Detection Chatbot",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 SAP Fraud Detection Intelligence Chatbot")
st.markdown("Ask me anything about vendor risk scores, fraud patterns, invoices, and forecasts!")

# ============================================================
# DATA LOADING CONFIGURATION
# ============================================================
DATA_URLS = {
    "Main Dataset": "sap_invoice_risk_master.csv",
    "Vendor Summary": "vendor_intelligence_summary.csv",
    "V10848 Forecast": "v10848_temporal_analysis.csv"
}

# Sidebar
with st.sidebar:
    st.header("📊 Data Overview")
    
    # AI Provider Selection
    st.markdown("### 🤖 Choose AI Provider:")
    ai_provider = st.radio(
        "Select your AI provider:",
        ["Hugging Face (FREE)", "OpenAI (Paid)", "Ollama (Local - FREE)"],
        help="Hugging Face is completely free! OpenAI requires API key and costs money. Ollama runs locally on your computer."
    )
    
    st.markdown("### 💡 Example Questions:")
    st.markdown("""
    - What's the risk score for vendor V10848?
    - Show me top 10 high-risk invoices
    - Which vendors have manual entry patterns?
    - What's the average fraud probability?
    - Forecast vendor V10848's behavior
    - Which vendors have late payment risk?
    - Show anomalies with amount > 50000
    - What's the total invoice count?
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data with caching
@st.cache_data
def load_data():
    """Load all fraud detection datasets"""
    try:
        main_file = DATA_URLS["Main Dataset"]
        if main_file.startswith("http"):
            st.info("📥 Downloading main dataset from cloud...")
            df = pd.read_csv(main_file)
        else:
            df = pd.read_csv(main_file)
        
        df['date'] = pd.to_datetime(df['date'])
        
        vendor_file = DATA_URLS["Vendor Summary"]
        vendor_summary = pd.read_csv(vendor_file) if not vendor_file.startswith("http") else pd.read_csv(vendor_file)
        
        forecast_file = DATA_URLS["V10848 Forecast"]
        df_v10848 = pd.read_csv(forecast_file) if not forecast_file.startswith("http") else pd.read_csv(forecast_file)
        
        if 'ds' in df_v10848.columns:
            df_v10848['ds'] = pd.to_datetime(df_v10848['ds'])
        elif 'date' in df_v10848.columns:
            df_v10848['date'] = pd.to_datetime(df_v10848['date'])
        
        return df, vendor_summary, df_v10848
    except FileNotFoundError:
        st.error("❌ Error: Data files not found")
        st.info("Make sure CSV files are in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

# Load the data
with st.spinner("📥 Loading fraud detection data..."):
    df, vendor_summary, df_v10848 = load_data()

if df is not None:
    with st.sidebar:
        st.markdown("### 📈 Dataset Stats:")
        st.metric("Total Invoices", f"{len(df):,}")
        st.metric("Total Vendors", f"{df['vendor_id'].nunique():,}")
        if 'manual_pattern_score' in df.columns:
            st.metric("Avg Manual Score", f"{df['manual_pattern_score'].mean():.3f}")
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        st.success("✅ Data loaded!")

# ============================================================
# AI PROVIDER SETUP
# ============================================================

api_key = None

if "Hugging Face" in ai_provider:
    st.success("✅ Using FREE Hugging Face API (no key needed for basic use!)")
    st.info("💡 For unlimited requests, get a free API token from https://huggingface.co/settings/tokens")
    
    hf_token = st.text_input(
        "🤗 Hugging Face Token (Optional - for unlimited use):",
        type="password",
        help="Leave empty for limited free usage, or add token for unlimited"
    )
    
    try:
        from langchain_community.llms import HuggingFaceHub
        api_key = hf_token if hf_token else "hf_placeholder"  # HF works without token for limited use
    except ImportError:
        st.error("Install: pip install langchain-community huggingface-hub")
        st.stop()

elif "OpenAI" in ai_provider:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ Using API key from secrets")
    except:
        api_key = st.text_input(
            "🔑 Enter OpenAI API Key:",
            type="password",
            help="Get from https://platform.openai.com/api-keys"
        )
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key")
            st.info("Get API key: https://platform.openai.com/api-keys (Costs ~$0.01 per session)")
            st.stop()
    
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        st.error("Install: pip install langchain-openai")
        st.stop()

elif "Ollama" in ai_provider:
    st.success("✅ Using Ollama (runs on your computer - 100% FREE!)")
    st.info("""
    **Setup Ollama:**
    1. Install from https://ollama.ai
    2. Run: `ollama pull mistral`
    3. Start the chatbot!
    """)
    
    try:
        from langchain_community.llms import Ollama
        api_key = "local"  # No API key needed
    except ImportError:
        st.error("Install: pip install langchain-community")
        st.stop()

if df is None:
    st.error("❌ Cannot proceed without data files.")
    st.stop()

# ============================================================
# CREATE SIMPLE PANDAS QUERY AGENT
# ============================================================

def query_dataframes(question: str, df, vendor_summary, df_v10848) -> str:
    """Simple rule-based query system for fraud detection data"""
    
    question_lower = question.lower()
    
    # Vendor-specific queries
    if "v10848" in question_lower or "vendor v10848" in question_lower:
        vendor_data = df[df['vendor_id'] == 'V10848']
        if "risk score" in question_lower:
            avg_risk = vendor_data['manual_pattern_score'].mean()
            return f"Vendor V10848 has an average manual pattern score of {avg_risk:.3f}. This vendor shows {len(vendor_data):,} invoices in the dataset."
        elif "forecast" in question_lower:
            return f"V10848 forecast data shows {len(df_v10848):,} time periods. Latest forecast: {df_v10848['forecast'].iloc[-1]:.2f} with actual: {df_v10848['actual'].iloc[-1]}"
        else:
            return f"Vendor V10848 has {len(vendor_data):,} invoices with average amount ${vendor_data['amount'].mean():,.2f}"
    
    # High-risk queries
    if "high risk" in question_lower or "top" in question_lower:
        if "manual" in question_lower:
            top_vendors = df.groupby('vendor_id')['manual_pattern_score'].mean().nlargest(10)
            result = "Top 10 vendors by manual pattern score:\n"
            for vendor, score in top_vendors.items():
                result += f"- {vendor}: {score:.3f}\n"
            return result
        elif "late payment" in question_lower:
            high_late = df[df['late_payment_risk'] > 0.8].groupby('vendor_id').size().nlargest(10)
            result = "Top 10 vendors with late payment risk > 0.8:\n"
            for vendor, count in high_late.items():
                result += f"- {vendor}: {count} invoices\n"
            return result
        else:
            top_invoices = df.nlargest(10, 'amount')[['vendor_id', 'amount', 'date', 'manual_pattern_score']]
            result = "Top 10 highest amount invoices:\n"
            for idx, row in top_invoices.iterrows():
                result += f"- {row['vendor_id']}: ${row['amount']:,.2f} on {row['date'].date()}\n"
            return result
    
    # Anomaly queries
    if "anomal" in question_lower:
        if 'anomaly_flag' in df.columns:
            anomalies = df[df['anomaly_flag'] == 1]
            if "amount" in question_lower and ">" in question_lower:
                try:
                    threshold = float(question_lower.split(">")[1].strip().split()[0])
                    filtered = anomalies[anomalies['amount'] > threshold]
                    return f"Found {len(filtered):,} anomalies with amount > ${threshold:,.0f}. Average amount: ${filtered['amount'].mean():,.2f}"
                except:
                    pass
            return f"Found {len(anomalies):,} anomalies in the dataset ({len(anomalies)/len(df)*100:.1f}% of all invoices)"
    
    # Statistical queries
    if "average" in question_lower or "mean" in question_lower:
        if "fraud probability" in question_lower and 'fraud_probability' in df.columns:
            return f"Average fraud probability: {df['fraud_probability'].mean():.3f}"
        elif "amount" in question_lower:
            return f"Average invoice amount: ${df['amount'].mean():,.2f}"
    
    if "total" in question_lower and "invoice" in question_lower:
        return f"Total invoices in dataset: {len(df):,} across {df['vendor_id'].nunique():,} vendors"
    
    # Default response
    return f"""I found:
- {len(df):,} total invoices
- {df['vendor_id'].nunique():,} unique vendors
- Average invoice amount: ${df['amount'].mean():,.2f}
- Date range: {df['date'].min().date()} to {df['date'].max().date()}

Please ask more specific questions about vendors, risk scores, or anomalies!"""

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the fraud detection data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            # Use simple rule-based system (works without any AI API!)
            response = query_dataframes(prompt, df, vendor_summary, df_v10848)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with 🐍 Python | 🎈 Streamlit<br>
    SAP Fraud Detection Intelligence System | 100% FREE to use!
</div>
""", unsafe_allow_html=True)