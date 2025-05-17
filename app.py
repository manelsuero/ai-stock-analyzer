import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient
from ollama import Ollama

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

# — Sidebar: AI Summaries
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

    # --- Simple Moving Average (20)
    df["SMA20"] = df["Close"].rolling(20).mean()
    # --- Exponential Moving Average (20)
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    # --- Bollinger Bands (20, ±2σ)
    rolling_std = df["Close"].rolling(20).std()
    df["BB_up"]   = df["SMA20"] + 2 * rolling_std
    df["BB_down"] = df["SMA20"] - 2 * rolling_std
    # --- MACD (12,26) & Signal (9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]       = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # --- RSI (14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    st.subheader("Technical Indicators")
    st.line_chart(df[["SMA20","EMA20"]].dropna(), height=200)
    st.line_chart(df[["BB_up","Close","BB_down"]].dropna(), height=200)
    st.line_chart(df[["MACD","MACD_signal"]].dropna(), height=200)
    st.line_chart(df["RSI"].dropna(), height=200)

    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # ── 2. News Analysis ──────────────────────────────────────────────────
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    all_news = fh.general_news(category="general", min_id=0, date=datetime.now().strftime("%Y-%m-%d"))
    df_news = (
        pd.DataFrame(all_news)
          .assign(datetime=lambda d: pd.to_datetime(d.datetime, unit="s"))
          .query("datetime >= @start_date and datetime <= @end_date")
    )

    st.subheader("News Analysis")
    if df_news.empty:
        st.warning("No news found for that ticker/date range.")
    else:
        st.dataframe(df_news[["datetime","headline","source"]].head(max_articles))
        st.success("✅ News loaded. Next: AI summarization.")

        # ── 3. AI Summaries via Ollama ───────────────────────────────────────
        ollama = Ollama()
        summaries = []
        for headline in df_news["headline"].head(max_articles):
            resp = ollama.chat(
                model="llama2",
                prompt=(
                    "You are a concise financial analyst.\n\n"
                    "Summarize this headline in one sentence:\n\n"
                    f"\"{headline}\""
                )
            )
            summaries.append(resp.strip())

        summary_df = pd.DataFrame({
            "headline": df_news["headline"].head(max_articles),
            "summary":  summaries
        })
        st.subheader("AI News Summaries")
        st.table(summary_df)

        st.success("✅ AI summaries loaded.")
