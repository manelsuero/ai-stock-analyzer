import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# — 0. Page config
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# — Sidebar Inputs
with st.sidebar:
    ticker = st.text_input("Enter a stock ticker (e.g. AAPL)").upper().strip()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_date   = st.date_input("End Date",   datetime.today())
    if st.button("Analyze Stock"):
        run = True
    else:
        run = False

# Only run when they click
if run:
    # 1️⃣ Technical Indicators
    df = yf.download(ticker, start=start_date, end=end_date)

    # ←—— HERE’S THE FIX FOR THE KEYERRORS ——→
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error(f"No price data found for {ticker}.")
        st.stop()

    # Compute all your indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI"]   = (
        (df["Close"].diff().clip(lower=0).rolling(14).mean() /
         df["Close"].diff().abs().rolling(14).mean()
        ) * 100
    )
    macd_line = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = macd_line
    df["MACD_Signal"] = macd_line.ewm(span=9, adjust=False).mean()
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_STD"]    = df["Close"].rolling(20).std()
    df["BB_Upper"]  = df["BB_Middle"] + 2 * df["BB_STD"]
    df["BB_Lower"]  = df["BB_Middle"] - 2 * df["BB_STD"]

    # Drop any rows with missing
    df = df.dropna()

    st.subheader("1️⃣ Technical Indicators")
    st.line_chart(df[["Close", "SMA20", "EMA20"]], height=300)
    st.line_chart(df[["RSI"]], height=200)
    st.line_chart(df[["MACD", "MACD_Signal"]], height=200)
    st.line_chart(df[["BB_Upper", "BB_Middle", "BB_Lower"]], height=200)
    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # 2️⃣ News Analysis (unchanged from your working code)
    NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
    days = st.sidebar.slider("Days of news history", 1, 7, 3)
    max_articles = st.sidebar.slider("Max articles to fetch", 10, 100, 30)
    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={ticker}&from={from_date}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res = requests.get(url).json().get("articles", [])[:max_articles]
    if not res:
        st.warning("No news found for that ticker (or API limit).")
    else:
        news_df = pd.DataFrame([{
            "datetime": art["publishedAt"],
            "headline": art["title"],
            "source":   art["source"]["name"],
            "url":      art["url"]
        } for art in res])
        news_df["datetime"] = pd.to_datetime(news_df["datetime"])
        st.subheader("2️⃣ News Analysis")
        st.dataframe(news_df, use_container_width=True)
        st.success("✅ News Analysis loaded. Next: AI News Summaries.")

    # 3️⃣ AI-Powered News Summaries via Ollama
    st.subheader("3️⃣ AI News Summaries (via Ollama)")
    # (you’ll fill in your own Ollama prompt + client logic here)
    st.info("🔧 Summaries coming soon… wire up your Ollama client & prompt.")

