import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import praw
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ————————————— 0. CONFIG INICIAL —————————————
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ————————————— 1. INDICADORES TÉCNICOS —————————————
st.sidebar.header("1️⃣ Technical Indicators")
ticker = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("today"))

if st.sidebar.button("🔍 Analyze Stock"):
    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No market data found for that ticker.")
        st.stop()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # SMA20
    df["SMA20"] = df["Close"].rolling(20).mean()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.set_title(f"{ticker} RSI (14)")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.index, df["Close"], label="Close")
    ax2.plot(df.index, df["SMA20"], label="SMA20")
    ax2.set_title(f"{ticker} Close vs SMA20")
    ax2.legend(loc="upper left")
    st.pyplot(fig2)

    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # ————————————— 2. NEWS ANALYSIS (Finnhub) —————————————
    st.header("2️⃣ News Analysis")
    days_news = st.sidebar.slider("Days of news history", 1, 7, 3)
    max_news  = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

    # Credenciales Finnhub
    FINNHUB_KEY = st.secrets["FINNHUB_KEY"]
    now = datetime.utcnow()
    from_date = now - timedelta(days=days_news)
    url = (
        f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        f"&minId={int(from_date.timestamp())}"
    )
    resp = requests.get(url).json()
    news = resp[:max_news]

    if not news:
        st.warning("No news found for that ticker.")
    else:
        df_news = pd.DataFrame([{
            "datetime": datetime.fromtimestamp(item["datetime"]),
            "headline": item["headline"],
            "source": item["source"],
            "url": item["url"]
        } for item in news])
        st.dataframe(df_news, height=300)
        st.success("✅ News Analysis loaded. Next: Social Sentiment (Reddit).")

    # ————————————— 3. SOCIAL SENTIMENT (Reddit + VADER) —————————————
    st.header("3️⃣ Social Sentiment (Reddit)")
    days_red = st.sidebar.slider("Days of Reddit history", 1, 14, 7)
    max_red  = st.sidebar.slider("Max posts to fetch", 10, 200, 50)

    # Inicializar PRAW
    reddit = praw.Reddit(
        client_id=st.secrets["REDDIT_CLIENT_ID"],
        client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
        user_agent=st.secrets["REDDIT_USER_AGENT"]
    )

    # Recoger posts de /r/stocks y /r/investing
    cutoff = datetime.utcnow() - timedelta(days=days_red)
    analyzer = SentimentIntensityAnalyzer()
    posts = []
    for sub in ["stocks", "investing"]:
        for post in reddit.subreddit(sub).new(limit=max_red):
            if datetime.utcfromtimestamp(post.created_utc) >= cutoff and ticker in post.title.upper():
                vs = analyzer.polarity_scores(post.title)
                posts.append({
                    "datetime": datetime.utcfromtimestamp(post.created_utc),
                    "title": post.title,
                    "subreddit": sub,
                    "neg": vs["neg"],
                    "neu": vs["neu"],
                    "pos": vs["pos"],
                    "compound": vs["compound"],
                    "url": post.url
                })

    if not posts:
        st.warning("No Reddit posts found for that ticker.")
    else:
        df_red = pd.DataFrame(posts).sort_values("datetime", ascending=False)
        st.dataframe(df_red, height=300)

        # Gráfico de sentimiento compuesto en el tiempo
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(df_red["datetime"], df_red["compound"], marker="o", linestyle="-")
        ax3.set_ylabel("Compound Sentiment")
        ax3.set_title(f"{ticker} Reddit Sentiment over last {days_red} days")
        st.pyplot(fig3)

        st.success("✅ Social Sentiment loaded. All done!")

else:
    st.info("👈 Enter a ticker and click Analyze Stock to begin.")
