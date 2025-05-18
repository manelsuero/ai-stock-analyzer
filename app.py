# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 0. Configuración inicial ────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Sidebar: Market + News + Sentiment Options ────────────────────────
with st.sidebar.form("options"):
    st.header("🔢 Market & News Options")
    st.subheader("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

    st.markdown("---")
    st.subheader("📰 News Options")
    news_days = st.slider("Days of news history",  1, 7,   3, key="news_days")
    news_max  = st.slider("Max articles to fetch",10, 100, 30, key="news_max")

    st.markdown("---")
    st.subheader("💬 StockTwits Sentiment Options")
    st_tw_days = st.slider("Days of posts history",1, 14,  7, key="tw_days")
    st_tw_max  = st.slider("Max posts to fetch",   10, 200, 50, key="tw_max")

    analyze = st.form_submit_button("🔍 Analyze Stock")

# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Download & fundamental (técnico) indicators ──────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for “{ticker}” in that range.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# RSI14
delta     = df['Close'].diff()
gain      = delta.where(delta > 0, 0)
loss      = -delta.where(delta < 0, 0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

# MACD & Signal Line
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

st.header("1️⃣ Technical Indicators")

# RSI Plot
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI')
ax.legend(loc="upper left")
st.pyplot(fig)

# SMA vs Close
st.subheader("SMA20 vs Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close Price')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.set_ylabel('Price')
ax.legend(loc="upper left")
st.pyplot(fig)

# MACD & Signal
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'],        label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal Line')
ax.set_ylabel('Value')
ax.legend(loc="upper left")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ── 3. News Analysis via NewsAPI ────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={news_max}&"
        f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
        f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    r = requests.get(news_url, timeout=5).json()
    articles = r.get("articles", [])
    if not articles:
        st.warning("No news found (API limit or bad key).")
    else:
        df_news = pd.DataFrame([{
            "datetime": a["publishedAt"],
            "headline": a["title"],
            "source":   a["source"]["name"],
            "url":      a["url"]
        } for a in articles])
        df_news["datetime"] = pd.to_datetime(df_news["datetime"])
        st.dataframe(df_news, use_container_width=True)

st.success("✅ News Analysis loaded. Next: Social Sentiment (StockTwits).")
st.markdown("---")

# ── 4. Social Sentiment (StockTwits + Vader) ────────────────────────────
st.header("3️⃣ Social Sentiment (StockTwits)")
sia = SentimentIntensityAnalyzer()

@st.cache_data(ttl=3600)
def fetch_stocktwits(symbol, days, max_posts):
    end = int(pd.Timestamp.now().timestamp())
    start = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        msgs = requests.get(url, timeout=5).json().get("messages", [])[:max_posts]
    except Exception:
        msgs = []
    data = []
    for m in msgs:
        t = pd.to_datetime(m["created_at"])
        text = m.get("body","")
        cat = m.get("entities", {}).get("sentiment", {}).get("basic", None)
        comp = sia.polarity_scores(text)["compound"]
        data.append((t, text, cat, comp))
    return pd.DataFrame(data, columns=["date","text","cat_sent","score"])

df_tw = fetch_stocktwits(ticker, st_tw_days, st_tw_max)
if df_tw.empty:
    st.warning("No StockTwits posts found for that ticker.")
else:
    # daily avg compound
    daily = df_tw.set_index("date")["score"].resample("D").mean().fillna(0)
    st.line_chart(daily)
    st.markdown(f"_Avg compound sentiment over last {st_tw_days} days_")

    # breakdown Bull/Bear
    cat_counts = df_tw["cat_sent"].value_counts().reindex(["Bullish","Bearish"], fill_value=0)
    st.bar_chart(cat_counts)

    # top 5 extremes
    st.subheader("Top 5 posts (extreme sentiment)")
    extremes = df_tw.reindex(df_tw["score"].abs().sort_values(ascending=False).index).head(5)
    st.write(extremes[["date","text","cat_sent","score"]])

st.success("✅ Social Sentiment loaded.")
