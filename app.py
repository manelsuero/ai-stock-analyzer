import streamlit as st
import pandas as pd
import altair as alt
import praw
import re
import os
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Setup
st.set_page_config(page_title="📊 Stock Sentiment Analyzer", layout="wide")

# Download NLTK stuff
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
download_nltk()

# Load Reddit API
def connect_to_reddit():
    reddit = praw.Reddit(
        client_id=st.secrets["REDDIT_CLIENT_ID"],
        client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
        user_agent=st.secrets["REDDIT_USER_AGENT"]
    )
    reddit.read_only = True
    return reddit

# Clean and analyze text
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in tokens if word not in stop_words])

def score_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# Fetch Reddit posts using HOT instead of SEARCH (safer)
def fetch_reddit_posts(ticker, limit=50):
    reddit = connect_to_reddit()
    posts = []
    for submission in reddit.subreddit("all").hot(limit=200):
        if ticker.lower() in submission.title.lower():
            title = clean_text(submission.title)
            body = clean_text(submission.selftext)
            title_score = score_sentiment(title)
            body_score = score_sentiment(body)
            posts.append({
                "ticker": ticker,
                "title": submission.title,
                "cleaned_title": title,
                "title_compound": title_score["compound"],
                "created_utc": datetime.utcfromtimestamp(submission.created_utc),
                "upvotes": submission.score,
                "comments": submission.num_comments,
                "url": submission.url
            })
        if len(posts) >= limit:
            break
    return pd.DataFrame(posts)

# App layout
st.title("🧠 Reddit Stock Sentiment")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

if st.sidebar.button("🚀 Analyze"):
    st.info(f"Analyzing Reddit sentiment for **{ticker}**...")
    df = fetch_reddit_posts(ticker, limit=limit)

    if df.empty:
        st.error("❌ No Reddit posts found.")
    else:
        st.success("✅ Posts fetched successfully.")

        avg_score = df["title_compound"].mean()
        st.metric("Average Sentiment Score", f"{avg_score:.3f}")

        if avg_score > 0.05:
            st.markdown("**Overall Sentiment: 🟢 Positive**")
        elif avg_score < -0.05:
            st.markdown("**Overall Sentiment: 🔴 Negative**")
        else:
            st.markdown("**Overall Sentiment: 🟡 Neutral**")

        st.subheader("📊 Sentiment Distribution")
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Ratio": [
                (df["title_compound"] > 0.05).mean(),
                (df["title_compound"] < -0.05).mean(),
                ((df["title_compound"] >= -0.05) & (df["title_compound"] <= 0.05)).mean()
            ]
        })
        st.altair_chart(alt.Chart(sentiment_df).mark_bar().encode(
            x="Sentiment",
            y="Ratio",
            color="Sentiment"
        ).properties(width=600))

        st.subheader("📈 Sentiment Over Time")
        df["created_utc"] = pd.to_datetime(df["created_utc"])
        st.altair_chart(alt.Chart(df).mark_line().encode(
            x="created_utc:T",
            y="title_compound:Q"
        ).properties(width=900))

        st.subheader("📝 Top Reddit Posts")
        for _, row in df.head(10).iterrows():
            st.markdown(f"""
            **{row['created_utc'].strftime('%Y-%m-%d %H:%M')}**  
            Score: {row['title_compound']:.3f} | 👍 {row['upvotes']} | 💬 {row['comments']}  
            > {row['title']}  
            [🔗 View Post]({row['url']})
            """)

        st.download_button("📥 Download CSV", data=df.to_csv(), file_name=f"{ticker}_reddit_sentiment.csv")

