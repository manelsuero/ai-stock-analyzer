# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient
from ollama import Ollama  # make sure ollama is installed in your env

# ─── 0. PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Analyzer",
    layout="wide",
)

# ─── 1. SIDEBAR FORM ───────────────────────────────────────────────────
with st.sidebar.form(key="inputs_form"):
    ticker = st.text_input(
        "Enter a stock ticker (e.g. AAPL)",
        key="ticker",
        value=st.session_state.get("ticker", "AAPL"),
    )

    start_date = st.date_input(
        "Start Date",
        key="start_date",
        value=st.session_state.get("start_date", datetime.today() - timedelta(days=365)),
    )
    end_date = st.date_input(
        "End Date",
        key="end_date",
        value=st.session_state.get("end_date", datetime.today()),
    )

    days_of_news = st.slider(
        "Days of news history",
        1,
        7,
        key="days_of_news",
        value=st.session_state.get("days_of_news", 3),
    )
    max_articles = st.slider(
        "Max articles to fetch",
        10,
        100,
        key="max_articles",
        value=st.session_state.get("max_articles", 30),
    )

    analyze = st.form_submit_button("🔍 Analyze Stock")

# ─── 2. MAIN PROGRAM (runs only on submit) ──────────────────────────────
if analyze:
    # pull session_state values
    ticker = st.session_state.ticker.upper()
    sd = st.session_state.start_date
    ed = st.session_state.end_date
    dn = st.session_state.days_of_news
    ma = st.session_state.max_articles

    st.title("📈 AI Stock Analyzer")
    st.success(f"Running analysis for **{ticker}** from {sd} → {ed}")

    # 2.1 Fetch price data
    df = yf.download(ticker, start=sd, end=ed)
    if df.empty:
        st.error(f"No market data found for ticker {ticker}.")
        st.stop()

    # 2.2 Compute technical indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    # 2.3 Display indicators
    st.header("1️⃣ Technical Indicators")
    st.line_chart(df[["Close", "SMA20", "EMA20"]], height=300)
    st.line_chart(df[["RSI"]], height=200)
    st.line_chart(df[["BB_Upper", "BB_Middle", "BB_Lower"]], height=200)

    # 2.4 News via Finnhub
    st.header("2️⃣ News Analysis")
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    now = int(datetime.now().timestamp())
    past = int((datetime.now() - timedelta(days=dn)).timestamp())
    try:
        news = fh.general_news(category="general", min_id=None)
        recent = [n for n in news if past <= n["datetime"] <= now][:ma]
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
        st.error("Error fetching news; check your FINNHUB_KEY in Secrets.")

    # 2.5 AI Summaries via Ollama
    st.header("3️⃣ AI News Summaries (via Ollama)")
    try:
        oll = Ollama()  # adjust constructor to your setup
        summaries = []
        for article in recent:
            prompt = (
                f"Summarize this news headline in one sentence:\n\n"
                f"{article['headline']}\n\n"
            )
            resp = oll.completion(model="llama2", prompt=prompt)
            summaries.append({
                "Headline": article["headline"],
                "Summary": resp["choices"][0]["message"]["content"].strip()
            })
        df_sum = pd.DataFrame(summaries)
        st.dataframe(df_sum)
    except Exception:
        st.error("Error running Ollama; ensure the Python `ollama` package is installed and your local Ollama daemon is running.")

