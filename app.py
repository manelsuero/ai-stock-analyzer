import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import finnhub

# ── Página ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar con formulario ──────────────────────────────────────────────────────
with st.sidebar.form("options_form"):
    st.header("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

    st.markdown("---")
    st.header("📰 News Options")
    news_days  = st.slider("Days of news history",    1, 7, 3)
    news_max   = st.slider("Max articles to fetch", 10, 100, 30)

    analyze_btn = st.form_submit_button("🔍 Analyze Stock")

# ── Sólo al pulsar el botón ──────────────────────────────────────────────────────
if not analyze_btn:
    st.info("👈 Fill in the options and click **Analyze Stock**")
    st.stop()

# ── 1️⃣ Descargar datos y calcular indicadores técnicos ─────────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for “{ticker}”.")
    st.stop()

# SMA20
df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
# RSI 14
delta     = df["Close"].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df["RSI"] = 100 - 100 / (1 + avg_gain/avg_loss)
# MACD + Señal
ema12           = df["Close"].ewm(span=12, adjust=False).mean()
ema26           = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]      = ema12 - ema26
df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()

# ── Mostrar indicadores ────────────────────────────────────────────────────────
st.markdown("## 1️⃣ Technical Indicators")

# RSI
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df["RSI"], label="RSI")
ax.set_ylabel("RSI")
ax.legend()
st.pyplot(fig)

# SMA20
st.subheader("SMA20 vs Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df["Close"], label="Close")
ax.plot(df.index, df["SMA20"], label="SMA20")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# MACD
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df["MACD"], label="MACD")
ax.plot(df.index, df["Signal"], label="Signal")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.success("✅ Technical indicators loaded. Next: News Analysis.")

# ── 2️⃣ News Analysis via Finnhub ───────────────────────────────────────────────
st.markdown("## 2️⃣ News Analysis")

# Chequea API key
finnhub_key = st.secrets.get("FINNHUB_KEY", "")
if not finnhub_key:
    st.error("🗝️ Please set your `FINNHUB_KEY` in Streamlit Secrets (no `[general]` header).")
    st.stop()

client = finnhub.Client(api_key=finnhub_key)

# Fechas en Unix
end_ts   = int(pd.Timestamp.today().timestamp())
start_ts = int((pd.Timestamp.today() - pd.Timedelta(days=news_days)).timestamp())

try:
    news = client.general_news(symbol=ticker.upper(), _from=pd.Timestamp(start_ts, unit="s").isoformat(),
                               to=pd.Timestamp(end_ts,   unit="s").isoformat())[:news_max]
except Exception as e:
    st.error(f"Error fetching news: {e}")
    st.stop()

if not news:
    st.warning("No news found for that ticker.")
else:
    # Construye DataFrame
    rows = []
    for item in news:
        rows.append({
            "datetime": pd.to_datetime(item["datetime"], unit="s"),
            "headline": item["headline"],
            "source":   item.get("source"),
            "url":      item.get("url")
        })
    df_news = pd.DataFrame(rows).set_index("datetime")
    st.dataframe(df_news, use_container_width=True)

st.markdown("---")
st.info("✅ News loaded. Next: Social Media Sentiment (coming soon).")
