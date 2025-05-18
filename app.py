# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from finnhub import Client as FinnhubClient

# ─── 0. CONFIG PAGE ──────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

# ─── 1. SIDEBAR FORM ─────────────────────────────────────────────────────
with st.sidebar.form("options_form"):
    ticker    = st.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL").upper()
    start_dt  = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_dt    = st.date_input("End Date",   datetime.today())
    days_news = st.slider("Days of news history",   1, 7, 3)
    max_news  = st.slider("Max articles to fetch", 10, 100, 30)
    analyze   = st.form_submit_button("🔍 Analyze Stock")

if not analyze:
    st.title("📈 AI Stock Analyzer")
    st.info("Use the sidebar to choose ticker, fechas y noticias, luego haz click en Analyze Stock.")
    st.stop()

# ─── 2. MARKET DATA & TECHNICAL INDICATORS ───────────────────────────────
st.title("📈 AI Stock Analyzer")
st.success(f"Running analysis for **{ticker}** from {start_dt} → {end_dt}")

# 2.1 Download
df = yf.download(ticker, start=start_dt, end=end_dt)

if df.empty:
    st.error(f"No market data found for {ticker} en ese rango.")
    st.stop()

# 2.2 Flatten multi‐index columns (yfinance a veces devuelve multi‐nivel)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)

# 2.3 Si no hay “Close”, abortamos
if "Close" not in df.columns:
    st.error("La serie 'Close' no está presente en los datos descargados.")
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

# Wrap each chart en try para no romper si hay muy pocos datos
try:
    st.line_chart(df[["Close","SMA20","EMA20"]], height=300)
except KeyError:
    st.warning("No hay suficientes datos para SMA/EMA.")

try:
    st.line_chart(df[["RSI"]], height=200)
except KeyError:
    st.warning("No hay suficientes datos para RSI.")

try:
    st.line_chart(df[["BB_Upper","BB_Mid","BB_Lower"]], height=200)
except KeyError:
    st.warning("No hay suficientes datos para Bollinger Bands.")

# ─── 3. NEWS ANALYSIS ─────────────────────────────────────────────────────
st.header("2️⃣ News Analysis")
try:
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    now_ts  = int(datetime.now().timestamp())
    past_ts = int((datetime.now() - timedelta(days=days_news)).timestamp())

    all_news = fh.general_news("general", min_id=None)
    filtered = [
        n for n in all_news
        if past_ts <= n.get("datetime", 0) <= now_ts
    ][:max_news]

    if not filtered:
        st.warning("No news found for that ticker / periodo.")
    else:
        df_news = pd.DataFrame([{
            "Date":     datetime.fromtimestamp(n["datetime"]),
            "Headline": n["headline"],
            "Source":   n["source"],
            "URL":      n["url"]
        } for n in filtered])
        st.dataframe(df_news)

except Exception:
    st.error("Error fetching news — revisa tu FINNHUB_KEY en Secrets.")

# ─── 4. AI News Summaries (Punto 3) ───────────────────────────────────────
st.header("3️⃣ AI News Summaries")
st.info("🚧 Aquí integraremos la IA (Ollama, OpenAI, etc.) para generar resúmenes.")

