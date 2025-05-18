import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIGURACIÓN ─────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ─── 1. SIDEBAR FORM ─────────────────────────────────────────────────────
with st.sidebar.form("options_form"):
    ticker    = st.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL").upper()
    start_dt  = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_dt    = st.date_input("End Date",   datetime.today())
    days_news = st.slider("Days of news history",   1, 7, 3)
    max_news  = st.slider("Max articles to fetch", 10, 100, 30)
    analyze   = st.form_submit_button("🔍 Analyze Stock")
# ── 1. Sidebar: Market + News + Sentiment Options ────────────────────────
with st.sidebar.form("options"):
    st.header("🔢 Market & News Options")
    st.subheader("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

if not analyze:
    st.title("📈 AI Stock Analyzer")
    st.info("Use the sidebar to choose ticker, fechas y noticias, luego haz click en Analyze Stock.")
    st.stop()
    st.markdown("---")
    st.subheader("📰 News Options")
    news_days = st.slider("Days of news history",  1, 7,   3, key="news_days")
    news_max  = st.slider("Max articles to fetch",10, 100, 30, key="news_max")

# ─── 2. MARKET DATA & TECHNICAL INDICATORS ───────────────────────────────
st.title("📈 AI Stock Analyzer")
st.success(f"Running analysis for **{ticker}** from {start_dt} → {end_dt}")
    st.markdown("---")
    st.subheader("💬 StockTwits Sentiment Options")
    st_tw_days = st.slider("Days of posts history",1, 14,  7, key="tw_days")
    st_tw_max  = st.slider("Max posts to fetch",   10, 200, 50, key="tw_max")

# 2.1 Download
df = yf.download(ticker, start=start_dt, end=end_dt)
    analyze = st.form_submit_button("🔍 Analyze Stock")

if df.empty:
    st.error(f"No market data found for {ticker} en ese rango.")
# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# 2.2 Flatten multi‐index columns (yfinance a veces devuelve multi‐nivel)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)

# 2.3 Si no hay “Close”, abortamos
if "Close" not in df.columns:
    st.error("La serie 'Close' no está presente en los datos descargados.")
# ── 2. Download & fundamental (técnico) indicators ──────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for “{ticker}” in that range.")
    st.stop()

# 2.4 Cálculo de indicadores
#  – SMA20, EMA20
df["SMA20"] = df["Close"].rolling(20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

#  – RSI
delta     = df["Close"].diff()
up        = delta.clip(lower=0)
down      = -delta.clip(upper=0)
roll_up   = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs        = roll_up / roll_down
df["RSI"] = 100 - (100 / (1 + rs))

#  – Bollinger Bands
df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_Std"]   = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

st.header("1️⃣ Technical Indicators")
# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# Wrap each chart en try para no romper si hay muy pocos datos
try:
    st.line_chart(df[["Close","SMA20","EMA20"]], height=300)
except KeyError:
    st.warning("No hay suficientes datos para SMA/EMA.")
# RSI14
delta     = df['Close'].diff()
gain      = delta.where(delta > 0, 0)
loss      = -delta.where(delta < 0, 0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

try:
    st.line_chart(df[["RSI"]], height=200)
except KeyError:
    st.warning("No hay suficientes datos para RSI.")
# MACD & Signal Line
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

try:
    st.line_chart(df[["BB_Upper","BB_Mid","BB_Lower"]], height=200)
except KeyError:
    st.warning("No hay suficientes datos para Bollinger Bands.")
st.header("1️⃣ Technical Indicators")


# ─── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)

# ─── FUNCIONES ────────────────────────────────────────────────
def fetch_news_sentiment(ticker, api_key, limit=50):
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&pageSize={limit}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        st.error(f"Error from NewsAPI: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    results = []

    for article in data.get("articles", []):
        title = article["title"]
        content = article["description"] or ""
        combined_text = f"{title} {content}"
        sentiment = analyzer.polarity_scores(combined_text)

        results.append({
            "title": title,
            "content": content,
            "published_at": article["publishedAt"],
            "source": article["source"]["name"],
            "url": article["url"],
            "sentiment_pos": sentiment["pos"],
            "sentiment_neg": sentiment["neg"],
            "sentiment_neu": sentiment["neu"],
            "sentiment_compound": sentiment["compound"]
        })

    df = pd.DataFrame(results)
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df

# ─── ANÁLISIS ──────────────────────────────────────────────────
if st.sidebar.button("🚀 Analyze"):
    st.info(f"Fetching and analyzing news about **{ticker}**...")
    df = fetch_news_sentiment(ticker, st.secrets["NEWSAPI_KEY"], limit)

    if not df.empty:
        st.success(f"Fetched and analyzed {len(df)} articles for **{ticker}**")

        # MÉTRICAS GENERALES
        avg_compound = df["sentiment_compound"].mean()
        st.metric("🧠 Average Sentiment Score", f"{avg_compound:.2f}")

        # VEREDICTO GENERAL
        pos = df["sentiment_pos"].mean()
        neg = df["sentiment_neg"].mean()
        neu = df["sentiment_neu"].mean()
        if pos > neg and pos > neu:
            verdict = "🟢 Positive"
        elif neg > pos and neg > neu:
            verdict = "🔴 Negative"
        else:
            verdict = "🟡 Neutral"
        st.markdown(f"### Overall Sentiment: {verdict}")

        # DISTRIBUCIÓN DE SENTIMIENTO
        st.subheader("📊 Sentiment Distribution")
        sentiment_dist = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Score": [pos, neg, neu]
        })
        chart = alt.Chart(sentiment_dist).mark_bar().encode(
            x=alt.X("Sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="Score",
            color=alt.Color("Sentiment", scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"],
                range=["#4CAF50", "#FFC107", "#F44336"]
            ))
        ).properties(width=700, height=300)
        st.altair_chart(chart)

        # TENDENCIA TEMPORAL
        st.subheader("📈 Sentiment Over Time")
        time_chart = alt.Chart(df).mark_line().encode(
            x=alt.X("published_at:T", title="Date"),
            y=alt.Y("sentiment_compound:Q", title="Compound Sentiment"),
            tooltip=["title", "sentiment_compound"]
        ).properties(width=900, height=400)
        st.altair_chart(time_chart)

        # TABLA DE NOTICIAS
        st.subheader("📰 News Table")
        st.dataframe(df[["published_at", "title", "sentiment_compound", "source", "url"]])

        # DESCARGA CSV
        st.download_button(
            "💾 Download CSV",
            df.to_csv(index=False),
            file_name=f"{ticker}_news_sentiment.csv"
        )
    else:
        st.warning("No data returned from NewsAPI.")
