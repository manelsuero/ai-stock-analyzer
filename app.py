import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import finnhub

# ── 0) Inicialización y configuración ──────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1) Sidebar & descarga datos ─────────────────────────────────────────────
st.sidebar.header("Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input( "Start Date", pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input(   "End Date",   pd.Timestamp.today())

if not st.sidebar.button("🔍 Analyze Stock"):
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No data for “{ticker}”.")
    st.stop()

# ── 2) Indicadores Técnicos ─────────────────────────────────────────────────
df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
delta = df["Close"].diff()
gain  = delta.where(delta>0, 0); loss = -delta.where(delta<0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df["RSI"] = 100 - (100/(1 + avg_gain/avg_loss))
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

st.header("🔍 Technical Indicators")
# — RSI
st.subheader("RSI (14 days)")
fig, ax = plt.subplots(); ax.plot(df.index, df["RSI"]); ax.set_ylabel("RSI")
st.pyplot(fig)
# — SMA20
st.subheader("SMA 20 over Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df["Close"], label="Close")
ax.plot(df.index, df["SMA20"], label="SMA20")
ax.set_ylabel("Price"); ax.legend()
st.pyplot(fig)
# — MACD
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df["MACD"], label="MACD")
ax.plot(df.index, df["Signal Line"], label="Signal")
ax.set_ylabel("Value"); ax.legend()
st.pyplot(fig)

st.markdown("---")
st.success("✅ Technical indicators loaded.")

# ── 3) News Analysis ─────────────────────────────────────────────────────────
st.header("📰 News Analysis")
# Inicializa cliente de Finnhub
fh = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])

# Trae las últimas 5 noticias para el ticker
news = fh.stock_news(symbol=ticker, category="general", min_id=None)
if not news:
    st.warning("No news found for that ticker.")
else:
    # convertimos a DataFrame y mostramos
    df_news = pd.DataFrame(news)[["datetime", "headline", "source", "url"]]
    df_news["datetime"] = pd.to_datetime(df_news["datetime"], unit="s")
    df_news = df_news.sort_values("datetime", ascending=False).head(5)
    for _, row in df_news.iterrows():
        st.markdown(
            f"**{row['headline']}**  \n"
            f"*{row['source']}  – {row['datetime'].strftime('%Y-%m-%d %H:%M')}*  \n"
            f"[Read more]({row['url']})"
        )

st.markdown("---")
st.success("✅ News loaded. Next: Social Media Sentiment (coming soon).")
