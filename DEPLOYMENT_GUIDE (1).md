# 🚀 Deploy SAP Fraud Detection Chatbot to Streamlit Cloud

## What You'll Get
A **live website** like `https://your-fraud-chatbot.streamlit.app` that business users can click and use instantly!

---

## 📋 Prerequisites

1. **GitHub Account** (free) - https://github.com/signup
2. **Streamlit Cloud Account** (free) - https://share.streamlit.io/signup
3. **OpenAI API Key** - https://platform.openai.com/api-keys
4. **Your 3 CSV files** downloaded from Databricks Volume

---

## 🎯 Deployment Steps

### STEP 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `sap-fraud-chatbot` (or any name)
3. Make it **Public** (required for free Streamlit Cloud)
4. Check ✅ "Add a README file"
5. Click **Create repository**

### STEP 2: Upload Files to GitHub

#### Option A: Using GitHub Web Interface (Easiest)

1. In your new repo, click **Add file** → **Upload files**
2. Drag and drop these files:
   ```
   fraud_detection_chatbot.py
   requirements.txt
   README.md
   .streamlit/secrets.toml  (we'll create this)
   ```
3. **DON'T upload the CSV files yet** (they're too large for GitHub)
4. Click **Commit changes**

#### Option B: Using Git Command Line

```bash
cd "D:\Python\AI FRAUD DETECTION PROJECT\CHATBOT"

git init
git add fraud_detection_chatbot.py requirements.txt README.md
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sap-fraud-chatbot.git
git push -u origin main
```

### STEP 3: Handle Large Data Files

**Problem**: GitHub has 100MB file limit. Your CSV files are too big!

**Solution**: Use one of these options:

#### 🔷 Option 1: GitHub Releases (Recommended)

1. In your GitHub repo, click **Releases** (right sidebar)
2. Click **Create a new release**
3. Tag: `v1.0`
4. Title: "Data Files"
5. Upload your 3 CSV files (up to 2GB each!)
6. Click **Publish release**
7. Right-click each file → **Copy link address**
8. Update the chatbot code to download from these URLs

#### 🔷 Option 2: Google Drive / Dropbox

1. Upload CSV files to Google Drive
2. Make them publicly accessible
3. Get shareable links
4. Use those links in the chatbot code

#### 🔷 Option 3: Use Sample Data (Fastest)

Create a smaller sample (10k rows) for demo:

```python
# Run this in Databricks to create small samples
df_small = df_export.sample(n=10000, random_state=42)
df_small.to_csv('/Volumes/workspace/default/financefraud/sap_invoice_risk_master_SMALL.csv', index=False)

vendor_summary.to_csv('/Volumes/workspace/default/financefraud/vendor_intelligence_summary_SMALL.csv', index=False)

df_v10848.head(100).to_csv('/Volumes/workspace/default/financefraud/v10848_temporal_analysis_SMALL.csv', index=False)
```

Then upload these SMALL files to GitHub directly.

### STEP 4: Create Secrets File for OpenAI Key

In your GitHub repo:

1. Create folder: `.streamlit`
2. Inside it, create file: `secrets.toml`
3. Add this content:

```toml
# .streamlit/secrets.toml
# NOTE: Do NOT put your actual key here if repo is public!
# This is just a placeholder

OPENAI_API_KEY = "sk-proj-..."
```

**IMPORTANT**: If your repo is public, DON'T put the real API key in GitHub! 
We'll add it in Streamlit Cloud settings instead (Step 6).

### STEP 5: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **New app**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/sap-fraud-chatbot`
   - **Branch**: `main`
   - **Main file path**: `fraud_detection_chatbot.py`
4. Click **Deploy!**

### STEP 6: Add Secrets in Streamlit Cloud

1. In Streamlit Cloud dashboard, click your app
2. Click ⚙️ **Settings** → **Secrets**
3. Add:

```toml
OPENAI_API_KEY = "sk-proj-YOUR-ACTUAL-KEY-HERE"
```

4. Click **Save**
5. App will auto-restart

### STEP 7: Update Chatbot Code to Use Secrets

Replace the API key input section with:

```python
import streamlit as st

# Try to get API key from secrets first (for deployed app)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fall back to user input (for local testing)
    api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key!")
        st.stop()
```

---

## ✅ Final Result

Your app will be live at:
```
https://YOUR_APP_NAME.streamlit.app
```

**Share this link** with business users - they just click and use! No installation needed.

---

## 🔧 Troubleshooting

### "File not found" error
- Make sure CSV files are in the same directory as `.py` file
- Or update code to use GitHub Release URLs

### "Module not found"
- Check `requirements.txt` has all packages
- Streamlit Cloud auto-installs from this file

### "API Key Invalid"
- Check secrets.toml format (no quotes around key)
- Make sure you saved secrets in Streamlit Cloud settings

### App is slow
- Large CSV files take time to load
- Consider using `@st.cache_data` decorator (already in code)
- Or use smaller sample files

---

## 🎨 Customization Ideas

1. **Custom Domain**: In Streamlit settings, add your company domain
2. **Password Protection**: Add `streamlit-authenticator` package
3. **Branding**: Update colors in `.streamlit/config.toml`
4. **Analytics**: Add Google Analytics tracking

---

## 💰 Costs

- **GitHub**: Free (public repos)
- **Streamlit Cloud**: Free (1 private app, unlimited public apps)
- **OpenAI API**: ~$0.15 per 1M tokens (GPT-4o-mini)
  - Typical chat session: $0.01
  - 100 users/day ≈ $30/month

---

## 📞 Support

If deployment fails:
1. Check Streamlit Cloud logs (bottom of app page)
2. Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud
3. Streamlit forum: https://discuss.streamlit.io

---

**Ready to deploy? Follow the steps above and your chatbot will be live in 10 minutes!** 🚀