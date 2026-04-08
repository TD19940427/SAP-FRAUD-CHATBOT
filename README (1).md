# 🔍 SAP Fraud Detection Intelligence Chatbot

An AI-powered chatbot that answers natural language questions about your SAP invoice fraud detection analysis using LangChain and Streamlit.

## 🌟 Features

* **Natural Language Interface**: Ask questions in plain English
* **Comprehensive Analysis**: Query 3M+ invoice records with fraud scores
* **Vendor Intelligence**: Get insights on vendor behavior patterns
* **Time Series Forecasts**: Explore ARIMA predictions for suspicious vendors
* **Risk Assessment**: Identify high-risk invoices, manual entry patterns, late payment risks

## 📋 What You Can Ask

### Vendor Analysis
* "What's the risk score for vendor V10848?"
* "Which vendors have the highest manual entry patterns?"
* "Show me vendors with late payment risk above 0.8"
* "What's the average fraud probability by vendor?"

### Invoice Investigation
* "Show me top 10 high-risk invoices"
* "Find anomalies with amount greater than 50,000"
* "What invoices have duplicate flags?"
* "Show invoices from vendor V10848 in the last month"

### Risk Metrics
* "What's the average fraud probability across all invoices?"
* "How many invoices are flagged as anomalies?"
* "Show distribution of risk scores"
* "Which scenarios have the highest fraud rates?"

### Time Series & Forecasts
* "Forecast vendor V10848's behavior for next week"
* "Show behavioral anomalies in V10848 forecast"
* "What's the forecast accuracy for V10848?"

## 🚀 Quick Start

### 1. Download Files from Databricks

#### Download Data Files (from Volume):
1. In Databricks, go to **Catalog** → **workspace** → **default** → **financefraud**
2. Download these 3 CSV files:
   - `sap_invoice_risk_master.csv` (~3M rows)
   - `vendor_intelligence_summary.csv` (1,000 vendors)
   - `v10848_temporal_analysis.csv` (ARIMA forecasts)

#### Download Application Files:
1. In Databricks, go to **Workspace** → **Users** → your email
2. Download:
   - `fraud_detection_chatbot.py`
   - `requirements.txt`
   - `README.md` (this file)

### 2. Setup Local Environment

Create a folder and place all 6 files together:
```
my-fraud-chatbot/
├── fraud_detection_chatbot.py
├── requirements.txt
├── README.md
├── sap_invoice_risk_master.csv
├── vendor_intelligence_summary.csv
└── v10848_temporal_analysis.csv
```

### 3. Install Dependencies

Open terminal in the folder and run:
```bash
pip install -r requirements.txt
```

### 4. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `sk-...`)

**Note**: OpenAI charges per API call. The chatbot uses `gpt-4o-mini` which is very affordable (~$0.15 per million tokens). A typical chat session costs < $0.01.

### 5. Run the Chatbot

```bash
streamlit run fraud_detection_chatbot.py
```

The app will open in your browser at `http://localhost:8501`

### 6. Start Chatting!

1. Enter your OpenAI API key in the text box
2. Type your question in the chat input
3. Get instant insights from your fraud detection data!

## 📊 Data Overview

### Main Dataset (sap_invoice_risk_master.csv)
* **3M+ invoices** with complete fraud analysis
* **Key columns**:
  - `vendor_id`, `invoice_id`, `amount`, `date`
  - `fraud_probability` - ML model prediction (0-1)
  - `anomaly_flag` - Isolation Forest detection
  - `final_risk_score` - Combined risk metric
  - `manual_pattern_score` - Manual entry detection
  - `late_payment_risk` - Payment delay probability
  - `discount_risk_score` - Discount capture risk
  - `benford_score` - Benford's Law compliance
  - `duplicate_flag` - Duplicate detection

### Vendor Summary (vendor_intelligence_summary.csv)
* **1,000 vendors** with aggregated statistics
* Average/max risk scores per vendor
* Total invoice counts and amounts
* Pattern scores across all risk categories

### V10848 Forecast (v10848_temporal_analysis.csv)
* **ARIMA(2,0,0)** time series forecast for suspicious vendor V10848
* Daily actual vs predicted invoice volumes
* Residuals and behavioral anomaly flags
* Used for temporal pattern detection

## 🛠️ Technical Architecture

* **Frontend**: Streamlit (interactive web UI)
* **AI Engine**: LangChain with OpenAI GPT-4o-mini
* **Data Processing**: Pandas DataFrames
* **Agent Type**: Pandas DataFrame Agent with OpenAI Functions

## 💡 Tips for Best Results

1. **Be Specific**: "Show top 5 vendors by risk score" works better than "tell me about vendors"
2. **Use Column Names**: The chatbot knows columns like `manual_pattern_score`, `late_payment_risk`
3. **Ask Follow-ups**: The chat history is preserved within the session
4. **Clear Chat**: Use the sidebar button to start fresh

## 🔒 Security & Privacy

* All data processing happens **locally** on your machine
* Only your questions and relevant data snippets are sent to OpenAI API
* CSV files never leave your computer
* API key is **never** stored - you enter it each session

## 🐛 Troubleshooting

### "Error loading data"
* Make sure all 3 CSV files are in the same folder as the Python script
* Check file names match exactly (case-sensitive)

### "Invalid API Key"
* Verify your OpenAI API key is correct
* Make sure you have credits in your OpenAI account
* Try regenerating the key at https://platform.openai.com/api-keys

### "Module not found"
* Run `pip install -r requirements.txt` again
* Try using a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## 📈 What's Next?

### Extend the Chatbot:
* Add more visualizations (charts, graphs)
* Export results to Excel/PDF
* Create scheduled email reports
* Add user authentication
* Deploy to Streamlit Cloud

### Enhance the Analysis:
* Integrate real-time SAP data
* Add more ML models
* Create alerting system
* Build dashboard widgets

## 📝 License

This is a demo project for educational and internal use. Make sure to comply with your organization's data policies when using actual SAP data.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit docs: https://docs.streamlit.io
3. Check LangChain docs: https://python.langchain.com

---

**Built with** 🐍 Python | 🦜 LangChain | 🎈 Streamlit | 🤖 OpenAI

**Created**: April 2026 | **Version**: 1.0