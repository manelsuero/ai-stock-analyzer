import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
import praw
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources if not already installed
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

# ── 0. Initial Configuration ────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# Download NLTK resources at startup
download_nltk_resources()

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

# If Analyze button not clicked, stop here
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Download & Technical Indicators ──────────────────────────────────
with st.spinner("Downloading market data..."):
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

st.success("✅ Technical indicators loaded.")
st.markdown("---")

# ── 3. News Analysis via NewsAPI ────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
df_news = None

if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    with st.spinner("Fetching news articles..."):
        news_url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}&pageSize={news_max}&"
            f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
            f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        )
        try:
            r = requests.get(news_url, timeout=10).json()
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

st.success("✅ News Analysis loaded.")
st.markdown("---")

# ── 4. Social Media Sentiment (Reddit + VADER) ────────────────────────────
st.header("3️⃣ Social Media Sentiment")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    """Clean text by removing URLs, special chars, and stopwords"""
    # Safety check for None or non-string
    if not isinstance(text, str):
        return ""
        
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

# Function to fetch reddit posts with error handling
def fetch_reddit_posts(ticker, subreddits, limit=50, days=7):
    """Fetch Reddit posts for a ticker from specified subreddits"""
    posts = []
    
    try:
        # Initialize Reddit API connection
        reddit = praw.Reddit(
            client_id=st.secrets["REDDIT_CLIENT_ID"],
            client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
            user_agent=st.secrets["REDDIT_USER_AGENT"]
        )
        reddit.read_only = True
        
        # Calculate date limit
        date_limit = datetime.utcnow() - timedelta(days=days)
        
        # Search across specified subreddits
        subreddit_list = [s.strip() for s in subreddits.split(",")]
        subreddit_str = "+".join(subreddit_list)
        
        subreddit = reddit.subreddit(subreddit_str)
        
        for submission in subreddit.search(ticker, limit=limit):
            # Skip if post is older than our date limit
            post_date = datetime.utcfromtimestamp(submission.created_utc)
            if post_date < date_limit:
                continue
                
            # Clean and analyze text
            full_text = submission.title + " " + (submission.selftext or "")
            cleaned_text = clean_text(full_text)
            
            # Get sentiment scores
            sentiment = analyzer.polarity_scores(cleaned_text)
            
            posts.append({
                "date": post_date,
                "title": submission.title,
                "score": sentiment["compound"],
                "pos": sentiment["pos"],
                "neg": sentiment["neg"],
                "neu": sentiment["neu"],
                "upvotes": submission.score,
                "comments": submission.num_comments,
                "url": submission.url
            })
            
        return posts
    except Exception as e:
        st.error(f"Reddit API Error: {str(e)}")
        return []

# Fetch Reddit posts
with st.spinner(f"Analyzing Reddit sentiment for {ticker} across {subreddits}..."):
    posts = fetch_reddit_posts(ticker, subreddits, limit=reddit_max, days=reddit_days)

if posts:
    # Convert posts to DataFrame
    df_reddit = pd.DataFrame(posts)
    
    # Display sentiment statistics
    avg_sentiment = df_reddit["score"].mean()
    
    # Determine overall sentiment
    if avg_sentiment > 0.05:
        sentiment_text = "🟢 POSITIVE"
        sentiment_color = "green"
    elif avg_sentiment < -0.05:
        sentiment_text = "🔴 NEGATIVE"
        sentiment_color = "red"
    else:
        sentiment_text = "🟡 NEUTRAL"
        sentiment_color = "orange"
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Posts Analyzed", len(posts))
    with col2:
        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
    with col3:
        st.markdown(f"<h3 style='color:{sentiment_color};text-align:center'>{sentiment_text}</h3>", 
                   unsafe_allow_html=True)
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.hist(df_reddit["score"], bins=20, color='skyblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Posts")
    st.pyplot(fig)
    
    # Sentiment over time
    st.subheader("Sentiment Over Time")
    df_reddit = df_reddit.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(df_reddit["date"], df_reddit["score"], marker='o', linestyle='-', markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recent posts with sentiment
    st.subheader("Recent Reddit Posts")
    for _, row in df_reddit.head(10).iterrows():
        color = "green" if row["score"] > 0.05 else "red" if row["score"] < -0.05 else "gray"
        st.markdown(f"""
        **{row['date'].strftime('%Y-%m-%d %H:%M')}** · Score: <span style='color:{color}'><b>{row['score']:.3f}</b></span> · 👍 {row['upvotes']} · 💬 {row['comments']}
        
        > {row['title']}
        
        [View on Reddit]({row['url']})
        """, unsafe_allow_html=True)
else:
    st.error(f"No Reddit posts found for {ticker} in selected subreddits.")

st.markdown("---")
st.success("✅ Analysis complete!")
