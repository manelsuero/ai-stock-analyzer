import streamlit as st
import pandas as pd
import altair as alt
import os
import praw
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="📊 Stock Sentiment Dashboard", layout="wide")

# Download NLTK resources if not already installed
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

download_nltk_resources()

# Load API keys
def load_api_keys():
    secrets = st.secrets
    return {
        "REDDIT_CLIENT_ID": secrets["REDDIT_CLIENT_ID"],
        "REDDIT_SECRET": secrets["REDDIT_CLIENT_SECRET"],
        "REDDIT_USER_AGENT": secrets["REDDIT_USER_AGENT"]
    }

def connect_to_reddit():
    keys = load_api_keys()
    reddit = praw.Reddit(
        client_id=keys["REDDIT_CLIENT_ID"],
        client_secret=keys["REDDIT_SECRET"],
        user_agent=keys["REDDIT_USER_AGENT"]
    )
    return reddit

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def score_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def fetch_reddit_posts(ticker, limit=100):
    reddit = connect_to_reddit()
    subreddit = reddit.subreddit("all")
    posts = []

    for submission in subreddit.search(ticker, limit=limit):
        cleaned_title = clean_text(submission.title)
        cleaned_body = clean_text(submission.selftext)

        title_scores = score_sentiment(cleaned_title)
        body_scores = score_sentiment(cleaned_body)

        posts.append({
            "ticker": ticker,
            "title": submission.title,
            "cleaned_title": cleaned_title,
            "title_pos": title_scores["pos"],
            "title_neg": title_scores["neg"],
            "title_neu": title_scores["neu"],
            "title_compound": title_scores["compound"],
            "selftext": submission.selftext,
            "cleaned_body": cleaned_body,
            "body_pos": body_scores["pos"],
            "body_neg": body_scores["neg"],
            "body_neu": body_scores["neu"],
            "body_compound": body_scores["compound"],
            "created_utc": datetime.utcfromtimestamp(submission.created_utc),
            "upvotes": submission.score,
            "num_comments": submission.num_comments,
            "url": submission.url
        })

    df = pd.DataFrame(posts)
    if not os.path.exists("data/reddit"):
        os.makedirs("data/reddit")
    file_path = f"data/reddit/{ticker}_reddit_posts_cleaned.csv"
    df.to_csv(file_path, index=False)
    return df

# Streamlit app
st.set_page_config(page_title="📊 Stock Sentiment Dashboard", layout="wide")
st.title("📊 Real-Time Reddit Sentiment")

st.sidebar.title("📥 Sentiment Analyzer")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

if st.sidebar.button("🚀 Analyze"):
    st.info(f"Fetching and analyzing Reddit posts for **{ticker}**...")
    data = fetch_reddit_posts(ticker, limit=limit)

    if data is not None:
        st.success(f"Data for **{ticker}** successfully loaded!")

        st.subheader(f"📊 Sentiment Overview for **{ticker}**")
        avg_compound = data["title_compound"].mean()
        sentiment_counts = data[["title_pos", "title_neg", "title_neu"]].mean()

        if sentiment_counts["title_pos"] > sentiment_counts["title_neg"] and sentiment_counts["title_pos"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🟢 Positive"
        elif sentiment_counts["title_neg"] > sentiment_counts["title_pos"] and sentiment_counts["title_neg"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🔴 Negative"
        else:
            sentiment_verdict = "🟡 Neutral"

        st.metric(label="Average Sentiment Score", value=f"{avg_compound:.2f}")
        st.markdown(f"**Overall Sentiment:** {sentiment_verdict}")

        st.subheader("📊 Sentiment Distribution")
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Ratio": [sentiment_counts["title_pos"], sentiment_counts["title_neg"], sentiment_counts["title_neu"]]
        })
        bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X("Sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="Ratio",
            color=alt.Color("Sentiment", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#4CAF50", "#FFC107", "#F44336"]))
        ).properties(width=700, height=300)
        st.altair_chart(bar_chart)

        st.subheader("📈 Sentiment Trend Over Time")
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        sentiment_line = alt.Chart(data).mark_line().encode(
            x=alt.X('created_utc:T', title="Date"),
            y=alt.Y('title_compound:Q', title="Sentiment Score"),
            color=alt.value("#4A90E2")
        ).properties(width=900, height=400)
        st.altair_chart(sentiment_line)

        st.subheader("💬 Recent Reddit Posts")
        for _, row in data.head(10).iterrows():
            st.markdown(f"""
            **{row['created_utc'].strftime('%Y-%m-%d %H:%M')}** · Score: {row['title_compound']:.3f} · 👍 {row['upvotes']} · 💬 {row['num_comments']}  
            > {row['title']}  
            [🔗 View Post]({row['url']})
            """)

        # Download
        st.download_button(label="💾 Download CSV", data=data.to_csv(), file_name=f"{ticker}_reddit_posts_cleaned.csv")
    else:
        st.error("❌ Failed to load data. Try again.")
