# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ─── 0. CONFIG & TITLE ───────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ─── 1. SIDEBAR: Market Data Options ──────────────────────────────────────────
st.sidebar.header("🔧 Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())
analyze_button = st.sidebar.button("🔍 Analyze Stock")

if not analyze_button:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ─── 2. DOWNLOAD & TECHNICAL INDICATORS ──────────────────────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No data found for “{ticker}”.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# RSI (14)
delta     = df['Close'].diff()
gain      = delta.where(delta > 0, 0)
loss      = -delta.where(delta < 0, 0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + avg_gain/avg_loss))

# MACD & Signal
ema12             = df['Close'].ewm(span=12, adjust=False).mean()
ema26             = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

st.success("✅ Technical indicators loaded.")

# ─── 3. PLOT TECHNICAL INDICATORS ───────────────────────────────────────────
st.header("🔍 Technical Indicators")

# RSI
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], color='tab:blue')
ax.set_ylabel('RSI')
st.pyplot(fig)

# SMA20 over Close
st.subheader("SMA 20 over Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close', linewidth=1)
ax.plot(df.index, df['SMA20'], label='SMA20', linewidth=1)
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# MACD & Signal
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'], label='MACD', linewidth=1)
ax.plot(df.index, df['Signal Line'], label='Signal', linewidth=1)
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

st.markdown("---")

# ─── 4. SIDEBAR: News Options ────────────────────────────────────────────────
st.sidebar.header("📰 News Options")
days_news    = st.sidebar.slider("Days of news history",    1, 7, 3)
max_articles = st.sidebar.slider("Max articles to fetch",  10, 100, 30)

# ─── 5. NEWS ANALYSIS via NewsAPI.org ────────────────────────────────────────
st.header("📰 News Analysis")

def fetch_news_newsapi(symbol, days, limit):
    api_key = st.secrets.get("NEWSAPI_KEY", "")
    if not api_key:
        st.error("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
        return pd.DataFrame()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        symbol,
        "from":     (pd.Timestamp.today() - pd.Timedelta(days=days)).date(),
        "sortBy":   "publishedAt",
        "pageSize": limit,
        "language": "en",
        "apiKey":   api_key
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.error(f"NewsAPI error [{resp.status_code}]")
        return pd.DataFrame()
    articles = resp.json().get("articles", [])
    data = []
    for art in articles:
        data.append({
            "publishedAt": pd.to_datetime(art["publishedAt"]),
            "source":      art["source"]["name"],
            "title":       art["title"],
            "url":         art["url"]
        })
    return pd.DataFrame(data)

df_news = fetch_news_newsapi(ticker, days_news, max_articles)
if df_news.empty:
    st.warning("No news found for that ticker (or API key issue).")
else:
    st.subheader("Latest News")
    # Mostrar tabla con enlace
    df_show = df_news.sort_values("publishedAt", ascending=False).reset_index(drop=True)
    df_show["link"] = df_show.apply(lambda r: f"[🔗]({r.url})", axis=1)
    st.dataframe(df_show[["publishedAt", "source", "title", "link"]])

# ─── 6. (Próximo) SOCIAL MEDIA SENTIMENT ────────────────────────────────────
# Aquí irá el bloque de Reddit/Stocktwits/etc. una vez estabilizado lo anterior.
