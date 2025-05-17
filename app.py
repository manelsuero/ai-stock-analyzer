import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient

# — 0. Page config
st.set_page_config(page_title="AI Stock Analyzer")
st.title("📈 AI Stock Analyzer")

# — Sidebar: Market Data Options
st.sidebar.header("1️⃣ Technical Indicators")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", "").upper()
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=90))
end_date   = st.sidebar.date_input("End Date", datetime.today())

# — Sidebar: News Options
st.sidebar.header("2️⃣ News Analysis")
days_of_news = st.sidebar.slider("Days of news history", 1, 7, 3)
max_articles = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

# — Sidebar: AI News Summaries
st.sidebar.header("3️⃣ AI News Summaries (via Ollama)")
if st.sidebar.button("🔍 Analyze Stock"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        st.stop()

    # ── 1. Technical Indicators ────────────────────────────────────────────
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No price data found for {ticker}.")
        st.stop()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    std20       = df["Close"].rolling(20).std()
    df["BB_up"]   = df["SMA20"] + 2*std20
    df["BB_down"] = df["SMA20"] - 2*std20
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100/(1+rs))

    st.subheader("Technical Indicators")
    st.line_chart(df[["SMA20","EMA20"]].dropna(), height=200)
    st.line_chart(df[["BB_up","Close","BB_down"]].dropna(), height=200)
    st.line_chart(df[["MACD","MACD_signal"]].dropna(), height=200)
    st.line_chart(df["RSI"].dropna(), height=200)
    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # ── 2. News Analysis ──────────────────────────────────────────────────
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    all_news = fh.general_news("general", min_id=0)
    df_news = (
        pd.DataFrame(all_news)
          .assign(datetime=lambda d: pd.to_datetime(d.datetime, unit="s"))
          .query("datetime >= @start_date and datetime <= @end_date")
    )

    st.subheader("News Analysis")
    if df_news.empty:
        st.warning("No news found for that ticker/date range.")
        st.stop()
    st.dataframe(df_news[["datetime","headline","source"]].head(max_articles))
    st.success("✅ News loaded. Next: AI summarization.")

    # ── 3. AI Summaries via Ollama HTTP API ───────────────────────────────
    def ollama_summary(headline: str) -> str:
        resp = requests.post(
            "http://127.0.0.1:11434/chat",
            json={
                "model": "llama2",
                "prompt": (
                    "You are a concise financial analyst.\n\n"
                    "Summarize this headline in one sentence:\n\n"
                    f"\"{headline}\""
                )
            },
            timeout=10
        )
        if resp.status_code != 200:
            return f"[Error {resp.status_code}]"
        return resp.json()["choices"][0]["message"]["content"].strip()

    summaries = [
        ollama_summary(h) for h in df_news["headline"].head(max_articles)
    ]
    summary_df = pd.DataFrame({
        "Headline": df_news["headline"].head(max_articles),
        "Summary":  summaries
    })

    st.subheader("AI News Summaries")
    st.table(summary_df)
    st.success("✅ AI summaries loaded.")
