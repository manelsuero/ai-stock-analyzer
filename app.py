import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import praw
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 0. Configuración inicial
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# 1. Indicadores Técnicos
st.sidebar.header("1️⃣ Technical Indicators")
ticker = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-04-15"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# News & Social parameters
days_news = st.sidebar.slider("Days of news history", 1, 7, 3)
max_news = st.sidebar.slider("Max articles to fetch", 10, 100, 30)
days_red = st.sidebar.slider("Days of Reddit history", 1, 14, 7)
max_red = st.sidebar.slider("Max Reddit posts", 10, 200, 50)

if st.sidebar.button("🔍 Analyze Stock"):
    # Fetch market data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No market data found for that ticker.")
        st.stop()

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # SMA20
    df['SMA20'] = df['Close'].rolling(20).mean()

    # Plot indicators
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.set_title(f"{ticker} RSI (14)")
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.index, df['Close'], label='Close')
    ax2.plot(df.index, df['SMA20'], label='SMA20')
    ax2.set_title(f"{ticker} Close vs SMA20")
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # 2. News Analysis (Finnhub)
    st.header("2️⃣ News Analysis")
    key = st.secrets.get("FINNHUB_KEY")
    if not key:
        st.error("Please set your FINNHUB_KEY in Streamlit Secrets.")
    else:
        now = datetime.utcnow()
        url = (
            f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
            f"&from={(now - timedelta(days=days_news)).date()}"
            f"&to={now.date()}&token={key}"
        )
        resp = requests.get(url).json()
        news = resp[:max_news]
        if not news:
            st.warning("No news found for that ticker.")
        else:
            df_news = pd.DataFrame([{
                'date': n['datetime'],
                'headline': n['headline'],
                'source': n['source'],
                'url': n['url']
            } for n in news])
            st.dataframe(df_news)
            st.success("✅ News Analysis loaded. Next: Social Sentiment (Reddit).")

    # 3. Social Sentiment (Reddit)
    st.header("3️⃣ Social Sentiment (Reddit)")
    cid = st.secrets.get("REDDIT_CLIENT_ID")
    secret = st.secrets.get("REDDIT_CLIENT_SECRET")
    agent = st.secrets.get("REDDIT_USER_AGENT")
    if not (cid and secret and agent):
        st.error("Please set your Reddit credentials in Streamlit Secrets.")
    else:
        reddit = praw.Reddit(
            client_id=cid,
            client_secret=secret,
            user_agent=agent
        )
        cutoff = datetime.utcnow() - timedelta(days=days_red)
        analyzer = SentimentIntensityAnalyzer()
        posts = []
        for sub in ['stocks', 'investing']:
            for post in reddit.subreddit(sub).new(limit=max_red):
                created = datetime.utcfromtimestamp(post.created_utc)
                if created >= cutoff and ticker in post.title.upper():
                    vs = analyzer.polarity_scores(post.title)
                    posts.append({
                        'datetime': created,
                        'title': post.title,
                        'subreddit': sub,
                        **vs,
                        'url': post.url
                    })
        if not posts:
            st.warning("No Reddit posts found for that ticker.")
        else:
            df_red = pd.DataFrame(posts).sort_values('datetime', ascending=False)
            st.dataframe(df_red)
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.plot(df_red['datetime'], df_red['compound'], marker='o')
            ax3.set_title(f"{ticker} Reddit Compound Sentiment")
            st.pyplot(fig3)
            st.success("✅ Social Sentiment loaded. All done!")
else:
    st.info("👈 Enter a ticker and click Analyze Stock to begin.")
