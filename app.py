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


# ── 4. Social Sentiment (Reddit + sentiment_analysis) ────────────────────
import requests
from datetime import datetime, timedelta
from sentiment_analysis import Analyzer  # pip install sentiment-analysis

st.header("3️⃣ Social Media Sentiment")

# Inicializar el analizador de la librería sentiment_analysis
sa = Analyzer()

def fetch_and_analyze_reddit(ticker, subreddits, days, max_posts):
    """
    1) Saca posts de Pushshift
    2) Analiza su sentimiento con sentiment_analysis
    """
    after_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    resumen = []

    for sub in [s.strip() for s in subreddits.split(",")]:
        url = "https://api.pushshift.io/reddit/search/submission"
        params = {
            "q":         ticker,
            "subreddit": sub,
            "after":     after_ts,
            "size":      max_posts
        }
        # peticion
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])

        for post in data:
            title = post.get("title", "")
            # analizar sentimiento
            score = sa.polarity_scores(title)["compound"]
            category = "Bullish" if score >= 0.05 else "Bearish" if score <= -0.05 else "Neutral"

            resumen.append({
                "subreddit": sub,
                "date":      datetime.fromtimestamp(post["created_utc"]),
                "title":     title,
                "url":       "https://reddit.com" + post.get("permalink", ""),
                "score":     score,
                "cat":       category
            })

    # ordenar cronológicamente
    return sorted(resumen, key=lambda x: x["date"], reverse=True)

with st.spinner("Fetching Reddit posts & sentiment..."):
    posts = fetch_and_analyze_reddit(ticker, subreddits, reddit_days, reddit_max)

if posts:
    # métricas generales
    scores = [p["score"] for p in posts]
    st.metric("Avg Sentiment", f"{sum(scores)/len(scores):.3f}")
    st.metric("Total Posts", len(posts))

    # grafico de dispersión
    dates = [p["date"] for p in posts]
    st.subheader("Sentiment over time")
    fig, ax = plt.subplots()
    ax.scatter(dates, scores, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("Sentiment score")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    # mostrar primeros 10
    st.subheader("Recent Reddit Posts with Sentiment")
    for p in posts[:10]:
        color = "green" if p["score"] >= 0.05 else "red" if p["score"] <= -0.05 else "gray"
        st.markdown(f"""
**r/{p['subreddit']}** · {p['date'].strftime('%Y-%m-%d %H:%M')} · **{p['cat']}**  
> {p['title']}  
[Ver en Reddit]({p['url']})
""", unsafe_allow_html=True)

else:
    st.warning("No Reddit posts found for this ticker/time period.")

st.success("✅ Social Sentiment Analysis loaded.")
st.markdown("---")
