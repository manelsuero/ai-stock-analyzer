import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
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
    st.subheader("💬 Reddit Sentiment Options")
    reddit_days = st.slider("Days of posts history", 1, 14, 7, key="reddit_days")
    reddit_max  = st.slider("Max posts to fetch", 10, 200, 50, key="reddit_max")
    subreddits  = st.text_input("Subreddits to search (comma separated)", 
                               value="stocks,investing,wallstreetbets")

    analyze = st.form_submit_button("🔍 Analyze Stock")

# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Download & fundamental (técnico) indicators ──────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for \"{ticker}\" in that range.")
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
ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)  # Overbought line
ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)  # Oversold line
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
ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
ax.legend(loc="upper left")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ── 3. News Analysis via NewsAPI ────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
df_news = None

if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={news_max}&"
        f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
        f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    try:
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
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")

st.success("✅ News Analysis loaded. Next: Social Sentiment (Reddit).")
st.markdown("---")


# ── 4. Social Sentiment (Reddit via PRAW only) ──────────────────────────
import praw
from datetime import datetime

st.header("3️⃣ Social Media Sentiment")

# Inicializa tu cliente PRAW (ya autenticado en st.secrets)
reddit = praw.Reddit(
    client_id     = st.secrets["REDDIT_CLIENT_ID"],
    client_secret = st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent    = st.secrets["REDDIT_USER_AGENT"]
)
reddit.read_only = True

# Función para obtener menciones o hot posts
def fetch_reddit_mentions_or_hot(ticker, subreddits, days, max_posts):
    results = []
    for sub in [s.strip() for s in subreddits.split(",")]:
        subreddit = reddit.subreddit(sub)

        # 1) Busca menciones exactas en el título
        query = f'title:"{ticker}"'
        mentions = list(subreddit.search(query,
                                         sort="new",
                                         time_filter="day" if days<=1 else "week" if days<=7 else "month",
                                         limit=max_posts))
        if mentions:
            for post in mentions:
                results.append((
                    post.created_utc,
                    sub,
                    post.title,
                    post.url
                ))
        else:
            # 2) Si no hay menciones, fallback a hot()
            hot = list(subreddit.hot(limit=5))
            for post in hot:
                results.append((
                    post.created_utc,
                    sub,
                    post.title,
                    post.url
                ))

    # Deduplicar por URL y ordenar por fecha descendente
    seen = set()
    unique = []
    for ts, sub, title, url in results:
        if url not in seen:
            seen.add(url)
            unique.append((ts, sub, title, url))

    unique.sort(key=lambda x: x[0], reverse=True)
    return unique

# Lanza la búsqueda
with st.spinner("Fetching Reddit posts via PRAW..."):
    posts = fetch_reddit_mentions_or_hot(
        ticker,
        subreddits,
        reddit_days,
        reddit_max
    )

# Mostrar
if posts:
    st.subheader("📝 Recent Reddit Posts")
    for ts, sub, title, url in posts[:10]:
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"""
**r/{sub}** · {dt}  
> {title}  
[Ver en Reddit]({url})
""")
else:
    st.warning("No se encontraron posts en los subreddits seleccionados.")

st.success("✅ Social Sentiment (Reddit via PRAW) loaded.")
st.markdown("---")
