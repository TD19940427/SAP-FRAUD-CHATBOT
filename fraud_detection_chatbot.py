import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# LangChain imports
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

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
# Option 1: Local files (for local testing)
# Option 2: GitHub Releases URLs (for cloud deployment)
# Option 3: Google Drive / Dropbox URLs

# UPDATE THESE URLs if using GitHub Releases or cloud storage:
DATA_URLS = {
    "Main Dataset": "sap_invoice_risk_master.csv",  # or "https://github.com/.../releases/download/v1.0/sap_invoice_risk_master.csv"
    "Vendor Summary": "vendor_intelligence_summary.csv",
    "V10848 Forecast": "v10848_temporal_analysis.csv"
}

# Sidebar for data info
with st.sidebar:
    st.header("📊 Data Overview")
    
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data with caching
@st.cache_data
def load_data():
    """Load all fraud detection datasets from local files or URLs"""
    try:
        # Main dataset
        main_file = DATA_URLS["Main Dataset"]
        if main_file.startswith("http"):
            st.info("📥 Downloading main dataset from cloud...")
            df = pd.read_csv(main_file)
        else:
            df = pd.read_csv(main_file)
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Vendor summary
        vendor_file = DATA_URLS["Vendor Summary"]
        if vendor_file.startswith("http"):
            vendor_summary = pd.read_csv(vendor_file)
        else:
            vendor_summary = pd.read_csv(vendor_file)
        
        # V10848 temporal analysis
        forecast_file = DATA_URLS["V10848 Forecast"]
        if forecast_file.startswith("http"):
            df_v10848 = pd.read_csv(forecast_file)
        else:
            df_v10848 = pd.read_csv(forecast_file)
        
        if 'ds' in df_v10848.columns:
            df_v10848['ds'] = pd.to_datetime(df_v10848['ds'])
        elif 'date' in df_v10848.columns:
            df_v10848['date'] = pd.to_datetime(df_v10848['date'])
        
        return df, vendor_summary, df_v10848
    except FileNotFoundError as e:
        st.error(f"❌ Error: Data files not found")
        st.info("""
        **For local deployment**: Make sure CSV files are in the same directory.
        
        **For cloud deployment**: 
        1. Upload CSV files to GitHub Releases
        2. Update DATA_URLS in the code with the download URLs
        
        See DEPLOYMENT_GUIDE.md for details.
        """)
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

# Load the data
with st.spinner("📥 Loading fraud detection data..."):
    df, vendor_summary, df_v10848 = load_data()

if df is not None:
    # Display data stats in sidebar
    with st.sidebar:
        st.markdown("### 📈 Dataset Stats:")
        st.metric("Total Invoices", f"{len(df):,}")
        st.metric("Total Vendors", f"{df['vendor_id'].nunique():,}")
        if 'final_risk_score' in df.columns:
            st.metric("Avg Risk Score", f"{df['final_risk_score'].mean():.3f}")
        elif 'manual_pattern_score' in df.columns:
            st.metric("Avg Manual Score", f"{df['manual_pattern_score'].mean():.3f}")
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        st.success("✅ Data loaded successfully!")

# ============================================================
# API KEY MANAGEMENT (Streamlit Cloud + Local)
# ============================================================

# Try to get API key from Streamlit Cloud secrets first
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("✅ Using API key from secrets")
except:
    # Fall back to user input for local testing
    api_key = st.text_input(
        "🔑 Enter your OpenAI API Key:", 
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to start chatting!")
        st.info("""
        **For local use**: Enter your key above
        
        **For cloud deployment**: Add to Streamlit Cloud secrets:
        1. Go to app settings → Secrets
        2. Add: `OPENAI_API_KEY = "sk-..."`
        
        Get API key: https://platform.openai.com/api-keys
        """)
        st.stop()

if df is None:
    st.error("❌ Cannot proceed without data files. Please check the configuration above.")
    st.stop()

# Initialize LangChain agent
@st.cache_resource
def create_agent(_df, _vendor_summary, _df_v10848, api_key):
    """Create LangChain pandas dataframe agent"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )
    
    # Create agent with all three dataframes
    agent = create_pandas_dataframe_agent(
        llm,
        [_df, _vendor_summary, _df_v10848],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        prefix="""
        You are a SAP fraud detection expert analyzing invoice data.
        
        You have access to THREE dataframes:
        - df (Main Dataset): Complete invoice dataset with risk scores, fraud probabilities, anomaly flags, manual patterns, etc.
          Key columns: vendor_id, amount, date, fraud_probability, anomaly_flag, final_risk_score, 
          manual_pattern_score, late_payment_risk, discount_risk_score, invoice_count, benford_score, etc.
        
        - vendor_summary (Vendor Intelligence): Aggregated vendor-level statistics
          Columns: vendor_id, avg_risk_score, max_risk_score, manual_pattern_score, late_payment_risk,
          discount_risk_score, total_invoices, total_amount, avg_amount
        
        - df_v10848 (Temporal Analysis): ARIMA forecast for vendor V10848
          Columns: date/ds, actual, forecast, residual, behavioral_anomaly
        
        When answering questions:
        - Be concise and specific
        - Show actual numbers and vendor IDs
        - For "top" queries, return 5-10 results
        - Format large numbers with commas
        - Explain risk scores (0-1 scale, higher = more risky)
        - Use the most relevant dataframe for each question
        """
    )
    return agent

try:
    agent = create_agent(df, vendor_summary, df_v10848, api_key)
except Exception as e:
    st.error(f"❌ Error creating AI agent: {e}")
    st.info("This might be an API key issue. Please check your OpenAI API key.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the fraud detection data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            response = agent.run(prompt, callbacks=[st_callback])
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
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
    Built with 🐍 Python | 🦜 LangChain | 🎈 Streamlit<br>
    SAP Fraud Detection Intelligence System
</div>
""", unsafe_allow_html=True)