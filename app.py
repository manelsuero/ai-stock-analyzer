# ── 1) IMPORTS & CONFIG ─────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 2) TÍTULO Y SIDEBAR ──────────────────────────────────────────────────────────
st.title("📈 AI Stock Analyzer")

st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

if st.sidebar.button("🔍 Analyze Stock"):

    # ── 3) ANÁLISIS TÉCNICO ────────────────────────────────────────────────
    st.header("📊 Technical Indicators")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data for {ticker}. Check the symbol.")
    else:
        # Indicadores
        df["SMA20"]   = ta.sma(df["Close"], length=20)
        df["RSI"]     = ta.rsi(df["Close"], length=14)
        macd         = ta.macd(df["Close"])
        df["MACD"]   = macd["MACD_12_26_9"]
        df["Signal"] = macd["MACDs_12_26_9"]

        # Mostrar gráficos
        st.subheader("RSI (14 days)")
        st.line_chart(df["RSI"])

        st.subheader("SMA 20 over Close Price")
        st.line_chart(df[["Close","SMA20"]])

        st.subheader("MACD & Signal Line")
        st.line_chart(df[["MACD","Signal"]])

    # ── 4) SENTIMIENTO EN REDDIT ──────────────────────────────────────────────
    st.header("💬 Reddit Sentiment")

    ## 4.1 Funciones de fetch y análisis
    sia = SentimentIntensityAnalyzer()

    def fetch_reddit_comments(ticker, after, before, size, pages):
        SUBS = ["stocks","investing","wallstreetbets","OptionsTrading","ValueInvesting"]
        all_comments = []
        params = {
            "q":         f"{ticker} OR ${ticker}",
            "subreddit": SUBS,
            "after":     after,
            "before":    before,
            "size":      size
        }
        for _ in range(pages):
            r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=params)
            batch = r.json().get("data", [])
            if not batch:
                break
            all_comments.extend(batch)
            params["before"] = batch[-1]["created_utc"]
        # filtrar longitud mínima
        filtered = [c for c in all_comments if len(c.get("body",""))>10]
        return pd.DataFrame({
            "date": pd.to_datetime([c["created_utc"] for c in filtered],unit="s"),
            "body": [c["body"] for c in filtered]
        })

    def analyze_sentiment(dfc):
        dfc["score"] = dfc["body"].apply(lambda t: sia.polarity_scores(t)["compound"])
        return dfc.set_index("date").resample("D").mean()["score"].fillna(method="ffill")

    @st.cache_data(ttl=3600)
    def get_reddit_sentiment(ticker, days, size, pages):
        end_unix   = int(pd.Timestamp.today().timestamp())
        start_unix = int((pd.Timestamp.today()-pd.Timedelta(days=days)).timestamp())
        dfc = fetch_reddit_comments(ticker, start_unix, end_unix, size, pages)
        st.sidebar.write(f"🔍 Fetched **{len(dfc)}** raw comments")
        if dfc.empty:
            return pd.Series(dtype=float)
        if len(dfc)>size:
            dfc = dfc.sample(size, random_state=42)
        return analyze_sentiment(dfc)

    ## 4.2 Sliders y ejecución
    days  = st.sidebar.slider("Days of history",           3, 30, 14)
    size  = st.sidebar.slider("Max comments to analyze",  50, 500, 300)
    pages = st.sidebar.slider("Pages of results to fetch", 1,   5,   3)

    sentiment = get_reddit_sentiment(ticker, days, size, pages)
    if sentiment.empty:
        st.info("No Reddit comments found for that ticker.")
    else:
        st.line_chart(sentiment)
        st.markdown(f"_Average daily sentiment over the last {days} days._")

    # ── 5) (Opcional) Próxima sección: News API & Ollama… ───────────────────
    st.success("Technical indicators & sentiment loaded! Next: News & AI Summary.")
