# app.py
import subprocess
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# 0️⃣ Page config
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

with st.sidebar:
    st.header("🔧 Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=365))
    end_date   = st.date_input("End Date",   value=datetime.today())

    st.markdown("---")
    st.header("📰 News Options")
    news_days    = st.slider("Days of news history", 1, 7, 3)
    max_articles = st.slider("Max articles to fetch", 10, 100, 30)

    st.markdown("---")
    st.header("🤖 AI News Summaries")
    run_all = st.button("🔍 Analyze Stock")

def summarize_with_ollama(texts: list[str]) -> str:
    prompt = (
        "Here are recent headlines and sources for a stock. "
        "Please summarize the main themes and overall sentiment.\n\n"
        + "\n\n".join(texts)
    )
    proc = subprocess.run(
        ["ollama", "run", "llama2", "--prompt", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()

if run_all:
    # 1️⃣ Technical Indicators
    st.header("1️⃣ Technical Indicators")
    df = yf.download(ticker, start=start_date, end=end_date)

    # ✂️ collapse MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("No price data found for that ticker.")
        st.stop()

    # Compute indicators on the full raw series
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta     = df["Close"].diff()
    up        = delta.clip(lower=0)
    down      = -delta.clip(upper=0)
    ma_up     = up.rolling(14).mean()
    ma_down   = down.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + ma_up/ma_down))

    # MACD
    exp1       = df["Close"].ewm(span=12, adjust=False).mean()
    exp2       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Up"]  = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]  = df["BB_Mid"] - 2 * df["BB_Std"]

    # now drop only rows where Close is missing
    df = df[df["Close"].notna()]

    # Chart
    st.line_chart(df[["Close","SMA20","EMA20"]], height=300)
    st.line_chart(df[["RSI"]], height=200)
    st.line_chart(df[["MACD","Signal"]], height=200)
    st.line_chart(df[["BB_Up","BB_Mid","BB_Lo"]], height=200)

    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # 2️⃣ News Analysis
    st.header("2️⃣ News Analysis")
    api_key = st.secrets["NEWSAPI_KEY"]
    since   = (datetime.utcnow() - timedelta(days=news_days)).strftime("%Y-%m-%d")
    url     = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&from={since}&sortBy=publishedAt"
        f"&pageSize={max_articles}&apiKey={api_key}"
    )
    resp     = requests.get(url).json()
    articles = resp.get("articles", [])
    if not articles:
        st.warning("No news found.")
        st.stop()

    df_news = pd.DataFrame([{
        "datetime": a["publishedAt"],
        "headline": a["title"],
        "source":   a["source"]["name"],
        "url":      a["url"]
    } for a in articles])
    st.dataframe(df_news, use_container_width=True)
    st.success("✅ News Analysis loaded. Next: AI Summary.")

    # 3️⃣ AI News Summaries via Ollama
    st.header("3️⃣ AI News Summaries (via Ollama)")
    texts = (df_news["headline"] + " — " + df_news["source"]).tolist()
    try:
        with st.spinner("Generating AI summary…"):
            summary = summarize_with_ollama(texts)
        st.subheader("📝 Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"Ollama failed: {e}")
