# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient

# ─── 0. PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

# ─── 1. SIDEBAR FORM ───────────────────────────────────────────────────
with st.sidebar.form("inputs"):
    ticker = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_date   = st.date_input("End Date", datetime.today())
    days_news  = st.slider("Days of news history", 1, 7, 3)
    max_news   = st.slider("Max articles to fetch", 10, 100, 30)
    analyze    = st.form_submit_button("🔍 Analyze Stock")

if not analyze:
    st.title("📈 AI Stock Analyzer")
    st.info("🔸 Use the sidebar to pick your ticker, dates, and news options, then click Analyze Stock.")
    st.stop()

# ─── 2. TECHNICAL INDICATORS ───────────────────────────────────────────
st.title("📈 AI Stock Analyzer")
st.success(f"Running analysis for **{ticker}** from {start_date} → {end_date}")

# Download market data
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for {ticker}. Double-check the ticker symbol or date range.")
    st.stop()

# Flatten multiindex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)

# Calculate the same indicators you had before:
df["SMA20"]    = df["Close"].rolling(20).mean()
df["EMA20"]    = df["Close"].ewm(span=20, adjust=False).mean()

delta          = df["Close"].diff()
up             = delta.clip(lower=0)
down           = -delta.clip(upper=0)
roll_up        = up.rolling(14).mean()
roll_down      = down.rolling(14).mean()
rs             = roll_up / roll_down
df["RSI"]      = 100 - (100 / (1 + rs))

df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_Std"]   = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

st.header("1️⃣ Technical Indicators")
st.line_chart(df[["Close","SMA20","EMA20"]], height=300)
st.line_chart(df[["RSI"]], height=200)
st.line_chart(df[["BB_Upper","BB_Mid","BB_Lower"]], height=200)

# ─── 3. NEWS ANALYSIS ─────────────────────────────────────────────────
st.header("2️⃣ News Analysis")
try:
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    now_ts  = int(datetime.now().timestamp())
    past_ts = int((datetime.now() - timedelta(days=days_news)).timestamp())

    all_news = fh.general_news("general", min_id=None)
    recent   = [
        n for n in all_news
        if past_ts <= n.get("datetime", 0) <= now_ts
    ][:max_news]

    if not recent:
        st.warning("No news found in that window.")
    else:
        df_news = pd.DataFrame([{
            "Date": datetime.fromtimestamp(n["datetime"]),
            "Headline": n["headline"],
            "Source":   n["source"],
            "URL":      n["url"]
        } for n in recent])
        st.dataframe(df_news)

except Exception:
    st.error("Error fetching news—please check your FINNHUB_KEY in Streamlit Secrets.")

# ─── 4. AI NEWS SUMMARIES ───────────────────────────────────────────────
st.header("3️⃣ AI News Summaries")
st.info("🔥 Coming soon — once Ollama’s import is stable, we’ll wire it in here.")
