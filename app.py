import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ── 0. Configuración de la página ─────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Formulario único para todas las opciones ────────────────────────────────
with st.sidebar.form("main_form"):
    st.markdown("## Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

    st.markdown("## News Options")
    days_news = st.slider("Days of news history", 1, 7, 3)
    max_news  = st.slider("Max articles to fetch", 10, 100, 30)

    st.markdown("## StockTwits Sentiment Options")
    days_tw = st.slider("Days of posts history", 1, 14, 7)
    max_tw  = st.slider("Max posts to fetch",    10, 200, 50)

    submit = st.form_submit_button("🔍 Analyze Stock")


if not submit:
    st.info("👈 Complete the form and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Validación básica ───────────────────────────────────────────────────────
ticker = ticker.strip().upper()
if not ticker:
    st.error("Ticker inválido.")
    st.stop()

# ── 3. Indicadores Técnicos ───────────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No hay datos para “{ticker}”.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
# RSI
delta = df['Close'].diff()
gain  = delta.where(delta > 0, 0)
loss  = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain/avg_loss)))
# MACD & Signal
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Plots
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI'); ax.legend(loc="upper left")
st.subheader("RSI (14 days)"); st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.set_ylabel('Price'); ax.legend(loc="upper left")
st.subheader("SMA20 & Close Price"); st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'], label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal')
ax.set_ylabel('Value'); ax.legend(loc="upper left")
st.subheader("MACD & Signal Line"); st.pyplot(fig)

st.success("✅ Technical indicators loaded.")

# ── 4. News Analysis ──────────────────────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"from={(pd.Timestamp.today() - pd.Timedelta(days=days_news)).date()}&"
        f"pageSize={max_news}&"
        f"apiKey={NEWSAPI_KEY}"
    )
    try:
        articles = requests.get(url, timeout=5).json().get("articles", [])
    except Exception:
        articles = []

    if not articles:
        st.warning("No news found for that ticker (or API issue).")
    else:
        df_news = pd.DataFrame([{
            "datetime": a["publishedAt"],
            "headline": a["title"],
            "source":   a["source"]["name"],
            "url":      a["url"]
        } for a in articles])
        df_news["datetime"] = pd.to_datetime(df_news["datetime"])
        st.dataframe(df_news, use_container_width=True)

st.success("✅ News Analysis loaded.")

# ── 5. Social Sentiment (StockTwits) ──────────────────────────────────────────
st.header("3️⃣ Social Sentiment (StockTwits)")

@st.cache_data(ttl=3600)
def fetch_stocktwits(sym: str, days: int, limit: int) -> pd.DataFrame:
    since = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json?limit={limit}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json().get("messages", [])
    except Exception:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "date": pd.to_datetime(m["created_at"]),
        "body": m.get("body","")
    } for m in data])
    return df

df_tw = fetch_stocktwits(ticker, days_tw, max_tw)
if df_tw.empty:
    st.warning("No StockTwits messages found for that ticker.")
else:
    counts = df_tw.set_index("date").resample("D").size()
    st.bar_chart(counts)
    st.markdown(f"_Total posts: {len(df_tw)} over last {days_tw} days._")
    st.dataframe(df_tw, use_container_width=True)

st.success("✅ Social Sentiment loaded.")
