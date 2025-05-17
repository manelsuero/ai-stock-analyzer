import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ── 0. Configuración inicial ────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

# ── Sidebar con FORM para inputs ────────────────────────────────────────────────
with st.sidebar.form("options_form"):
    st.header("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())
    st.markdown("---")
    st.header("News Options")
    days_news    = st.slider("Days of news history",    1, 7, 3)
    max_articles = st.slider("Max articles to fetch",  10,100,30)
    st.markdown("---")
    st.header("Social Options")
    days_soc = st.slider("Days of sentiment history",    1,14,7)
    # (aunque Finnhub ignora esta variable, la dejamos para UX)
    max_posts = st.slider("Max posts to fetch (unused)", 10,200,50)
    st.markdown("---")
    analyze = st.form_submit_button("🔍 Analyze Stock")

if not analyze:
    st.title("📈 AI Stock Analyzer")
    st.info("👈 Enter parameters in the sidebar and click **Analyze Stock**.")
    st.stop()

st.title("📈 AI Stock Analyzer")

# ── 1. Descarga datos y Technical Indicators ───────────────────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No data for “{ticker}”.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(20).mean()
# RSI
delta     = df['Close'].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100/(1 + avg_gain/avg_loss))
# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

st.header("1️⃣ Technical Indicators")
st.subheader("RSI (14 days)")
fig,ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.legend(); ax.set_ylabel("RSI")
st.pyplot(fig)

st.subheader("SMA20 vs Close Price")
fig,ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.legend(); ax.set_ylabel("Price")
st.pyplot(fig)

st.subheader("MACD & Signal Line")
fig,ax = plt.subplots()
ax.plot(df.index, df['MACD'],        label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal')
ax.legend(); ax.set_ylabel("Value")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis.")

# ── 2. News via Finnhub ────────────────────────────────────────────────────────
st.header("2️⃣ News Analysis")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "")
if not FINNHUB_KEY:
    st.warning("🔑 Please set your FINNHUB_KEY in Streamlit Secrets (without any [general] header).")
else:
    since = (pd.Timestamp.today() - pd.Timedelta(days=days_news)).date()
    url_news = (
        f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
        f"&from={since}&to={pd.Timestamp.today().date()}"
        f"&token={FINNHUB_KEY}"
    )
    try:
        news = requests.get(url_news, timeout=5).json()
    except Exception:
        news = []
    if not news:
        st.warning("No news found for that ticker.")
    else:
        df_news = pd.DataFrame(news).sort_values("datetime", ascending=False)
        df_news["datetime"] = pd.to_datetime(df_news["datetime"], unit="s")
        st.dataframe(df_news[["datetime","headline","source","url"]].head(max_articles))
        st.success("✅ News Analysis loaded. Next: Social Sentiment (Finnhub).")

# ── 3. Social Sentiment via Finnhub ────────────────────────────────────────────
st.header("3️⃣ Social Sentiment (Finnhub)")
if not FINNHUB_KEY:
    st.warning("🔑 Please set your FINNHUB_KEY in Streamlit Secrets.")
else:
    since = (pd.Timestamp.today() - pd.Timedelta(days=days_soc)).date()
    url_soc = (
        f"https://finnhub.io/api/v1/stock/social-sentiment?"
        f"symbol={ticker}&from={since}&to={pd.Timestamp.today().date()}"
        f"&token={FINNHUB_KEY}"
    )
    try:
        soc = requests.get(url_soc, timeout=5).json()
    except Exception:
        soc = {}
    tw = soc.get("twitter", [])
    rd = soc.get("reddit", [])

    if not tw and not rd:
        st.warning("No social sentiment data found for that ticker.")
    else:
        if tw:
            df_tw = pd.DataFrame(tw)
            df_tw["date"] = pd.to_datetime(df_tw["date"]).dt.date
            st.subheader("🐦 Twitter Mentions")
            st.line_chart(df_tw.set_index("date")["mention"])
            st.subheader("🐦 Twitter Sentiment")
            st.bar_chart (df_tw.set_index("date")["sentiment"])
        if rd:
            df_rd = pd.DataFrame(rd)
            df_rd["date"] = pd.to_datetime(df_rd["date"]).dt.date
            st.subheader("👥 Reddit Mentions")
            st.line_chart(df_rd.set_index("date")["mention"])
            st.subheader("👥 Reddit Sentiment")
            st.bar_chart (df_rd.set_index("date")["sentiment"])
        st.success("✅ Social Sentiment loaded via Finnhub.")
