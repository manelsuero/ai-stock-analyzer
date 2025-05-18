import praw
import os
import pandas as pd
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources if not already installed
nltk.download("punkt")
nltk.download("stopwords")

def load_api_keys():
    """
    Load API keys from the secrets file.
    """
    secrets_file = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_file):
        raise FileNotFoundError(f"🔴 Secrets file not found: {secrets_file}")
   
    with open(secrets_file, "r") as f:
        secrets = f.read()
   
    keys = {}
    for line in secrets.splitlines():
        if line.strip() and "=" in line:
            key, value = line.split("=", 1)
            keys[key.strip()] = value.strip().replace('"', '')

    return keys

def connect_to_reddit():
    """
    Connect to the Reddit API using credentials from the secrets file.
    """
    keys = load_api_keys()
   
    reddit = praw.Reddit(
        client_id=keys.get("REDDIT_CLIENT_ID"),
        client_secret=keys.get("REDDIT_SECRET"),
        user_agent="stock-predictor-bot/0.1 by your_reddit_username"
    )

    print("✅ Successfully connected to Reddit API")
    return reddit

def clean_text(text):
    """
    Clean the text by removing URLs, special characters, and stopwords.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
   
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
   
    # Tokenize the text
    tokens = word_tokenize(text.lower())
   
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
   
    # Join the tokens back into a single string
    cleaned_text = " ".join(cleaned_tokens)
   
    return cleaned_text

def score_sentiment(text):
    """
    Score the sentiment of the given text using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

def fetch_reddit_posts(ticker, limit=100):
    """
    Fetch recent Reddit posts mentioning the given stock ticker and score their sentiment.
    """
    reddit = connect_to_reddit()
    subreddit = reddit.subreddit("all")
    posts = []

    print(f"🔄 Fetching posts for {ticker}...")
    for submission in subreddit.search(ticker, limit=limit):
        cleaned_title = clean_text(submission.title)
        cleaned_body = clean_text(submission.selftext)
       
        # Score the sentiment
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
   
    # Convert to DataFrame
    df = pd.DataFrame(posts)
   
    # Create the data directory if it doesn't exist
    if not os.path.exists("data/reddit"):
        os.makedirs("data/reddit")
   
    # Save to CSV
    file_path = f"data/reddit/{ticker}_reddit_posts_cleaned.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Cleaned and scored data for {ticker} saved to {file_path}")

    return df

# Test the function
if __name__ == "__main__":
    fetch_reddit_posts("AAPL", limit=50)
    fetch_reddit_posts("TSLA", limit=50)


import streamlit as st
import pandas as pd
import altair as alt
from sentiment_analysis import fetch_reddit_posts
from data_processing import get_realtime_price_alpha, get_historical_price_alpha
import numpy as np

# Streamlit Page Configuration
st.set_page_config(page_title="📊 Real-Time Stock Sentiment Dashboard", layout="wide")

# Sidebar for input
st.sidebar.title("🔎 Stock Sentiment Analysis")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

# Fetch Data Button
if st.sidebar.button("🚀 Fetch and Analyze"):
    st.info(f"Fetching and analyzing Reddit posts for **{ticker}**...")
    data = fetch_reddit_posts(ticker, limit=limit)

    if data is not None:
        st.success(f"Data for **{ticker}** successfully loaded! ✅")

        # Sentiment Overview
        st.subheader(f"📊 Sentiment Overview for **{ticker}**")
        avg_compound = data["title_compound"].mean()
        sentiment_counts = data[["title_pos", "title_neg", "title_neu"]].mean()

        # Fixing the Sentiment Verdict Logic
        if sentiment_counts["title_pos"] > sentiment_counts["title_neg"] and sentiment_counts["title_pos"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🟢 Positive"
        elif sentiment_counts["title_neg"] > sentiment_counts["title_pos"] and sentiment_counts["title_neg"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🔴 Negative"
        else:
            sentiment_verdict = "🟡 Neutral"

        st.metric(label="Average Sentiment Score", value=f"{avg_compound:.2f}")
        st.markdown(f"**Overall Sentiment:** {sentiment_verdict}")

        # Sentiment Distribution
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Ratio": [sentiment_counts["title_pos"], sentiment_counts["title_neg"], sentiment_counts["title_neu"]]
        })
        st.subheader("📊 Sentiment Distribution")
        bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X("Sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="Ratio",
            color=alt.Color("Sentiment", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#4CAF50", "#FFC107", "#F44336"]))
        ).properties(width=700, height=300)
        st.altair_chart(bar_chart)

        # Sentiment Trend Line
        st.subheader("📈 Sentiment Trend Over Time")
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        sentiment_line = alt.Chart(data).mark_line().encode(
            x=alt.X('created_utc:T', title="Date"),
            y=alt.Y('title_compound:Q', title="Sentiment Score"),
            color=alt.value("#4A90E2")
        ).properties(width=900, height=400)
        st.altair_chart(sentiment_line)

        # Real-Time Stock Price
        current_price = get_realtime_price_alpha(ticker)
        st.subheader(f"💰 Real-Time Price for **{ticker}**: **${current_price}**")

                # Stock Price Over Time
        st.subheader(f"📉 Stock Price Trend for **{ticker}**")
        price_data = get_historical_price_alpha(ticker)

        if not price_data.empty:
            price_chart = alt.Chart(price_data).mark_line().encode(
                x=alt.X('Date:T', title="Date"),
                y=alt.Y('Close:Q', title="Stock Price ($)"),
                color=alt.value("#FFA500")
            ).properties(width=900, height=400)
            st.altair_chart(price_chart)

            # Correlation Analysis
            st.subheader("📊 Sentiment and Price Correlation")
            combined_df = pd.merge(data, price_data, left_on="created_utc", right_on="Date", how="inner")
            if not combined_df.empty:
                correlation = combined_df['title_compound'].corr(combined_df['Close'])
                st.metric(label="Correlation (Sentiment vs. Price)", value=f"{correlation:.2f}")
            else:
                st.warning("⚠️ No overlapping data for correlation.")
        else:
            st.warning("⚠️ No historical price data available for this ticker.")

        # Download Button
        data_file = f"data/reddit/{ticker}_reddit_posts_cleaned.csv"
        st.download_button(label="💾 Download Data as CSV", data=data.to_csv(), file_name=f"{ticker}_reddit_posts_cleaned.csv")
    else:
        st.error("❌ Failed to fetch data. Please check your ticker symbol and try again.")

# Footer
st.markdown("🔗 Created by **Albert Paradell** - Real-time stock analysis powered by Reddit sentiment.")
