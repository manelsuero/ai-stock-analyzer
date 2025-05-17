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
with st.sidebar.form(key="inputs"):
    ticker_input = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")

    start_input = st.date_input(
        "Start Date",
        value=datetime.today() - timedelta(days=365),
    )
    end_input = st.date_input(
        "End Date",
        value=datetime.today(),
    )

    days_news = st.slider("Days of news history", 1, 7, 3)
    max_news = st.slider("Max articles to fetch", 10, 100, 30)

    submit = st.form_submit_button("🔍 Analyze Stock")

# Only run the analysis when the user submits the form
if submit:
    ticker = ticker_input.upper()
    sd = start_input
    ed = end_input

    st.title("📈 AI Stock Analyzer")
    st.success(f"Running analysis for **{ticker}** from {sd} → {ed}")

    # ─── 2. TECHNICAL INDICATORS ────────────────────────────────────────
    df = yf.download(ticker, start=sd, end=ed)
    if df.empty:
        st.error(f"No market data for {ticker}. Check the ticker and date range.")
        st.stop()

    # Simple Moving / Exponential Moving
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    st.header("1️⃣ Technical Indicators")
    # prices + SMAs
    st.line_chart(df[["Close", "SMA20", "EMA20"]], height=300)
    # RSI
    st.line_chart(df[["RSI"]], height=200)
    # Bollinger
    st.line_chart(df[["BB_Upper", "BB_Middle", "BB_Lower"]], height=200)

    # ─── 3. NEWS ANALYSIS ───────────────────────────────────────────────
    st.header("2️⃣ News Analysis")
    try:
        fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
        now_ts = int(datetime.now().timestamp())
        past_ts = int((datetime.now() - timedelta(days=days_news)).timestamp())
        all_news = fh.general_news(category="general", min_id=None)
        recent = [
            n for n in all_news
            if past_ts <= n.get("datetime", 0) <= now_ts
        ][:max_news]

        if not recent:
            st.warning("No news found in that window.")
        else:
            df_news = pd.DataFrame([{
                "Date": datetime.fromtimestamp(n["datetime"]),
                "Headline": n["headline"],
                "Source": n["source"],
                "URL": n["url"],
            } for n in recent])
            st.dataframe(df_news)

    except Exception:
        st.error("Error fetching news. Check your FINNHUB_KEY in Secrets.")

    # ─── 4. AI SUMMARY PLACEHOLDER ────────────────────────────────────
    st.header("3️⃣ AI News Summaries")
    st.info("Ollama is temporarily disabled—will restore once the import is fixed.")

