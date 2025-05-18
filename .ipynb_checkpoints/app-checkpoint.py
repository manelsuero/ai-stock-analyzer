import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import praw
import os
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Descargar recursos de NLTK
nltk.download("punkt")
nltk.download("stopwords")

# ─── 🔐 Cargar claves de Reddit ─────────────────────────────────────────
def load_api_keys():
    secrets_file = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_file):
        raise FileNotFoundError(f"Secrets file not found: {secrets_file}")
    
    with open(secrets_file, "r") as f:
        secrets = f.read()
    
    keys = {}
    for line in secrets.splitlines():
        if line.strip() and "=" in line:
            key, value = line.split("=", 1)
            keys[key.strip()] = value.strip().replace('"', '')
    
    return keys

# ─── 🔗 Conexión con Reddit ─────────────────────────────────────────────
def connect_to_reddit():
    keys = load_api_keys()
    reddit = praw.Reddit(
        client_id=keys.get("REDDIT_CLIENT_ID"),
        client_secret=keys.get("REDDIT_SECRET"),
        user_agent="stock-sentiment-app"
    )
    return reddit

# ─── 🧹 Limpiar texto ───────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# ─── 🧠 Análisis de sentimiento ─────────────────────────────────────────
def score_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# ─── 📥 Obtener posts de Reddit ─────────────────────────────────────────
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

# ─── 💵 Simular precios ─────────────────────────────────────────────────
def get_realtime_price_alpha(ticker):
    return round(100 + np.random.randn(), 2)

def get_historical_price_alpha(ticker):
    dates = pd.date_range(end=datetime.today(), periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100
    return pd.DataFrame({"Date": dates, "Close": prices})

# ─── 📊 Streamlit App ───────────────────────────────────────────────────
st.set_page_config(page_title="📊 Real-Time Stock Sentiment Dashboard", layout="wide")
st.sidebar.title("🔎 Stock Sentiment Analysis")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

if st.sidebar.button("🚀 Fetch and Analyze"):
    st.info(f"Fetching and analyzing Reddit posts for **{ticker}**...")
    data = fetch_reddit_posts(ticker, limit=limit)

    if data is not None:
        st.success(f"Data for **{ticker}** successfully loaded! ✅")

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

        st.subheader("📈 Sentiment Trend Over Time")
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        sentiment_line = alt.Chart(data).mark_line().encode(
            x=alt.X('created_utc:T', title="Date"),
            y=alt.Y('title_compound:Q', title="Sentiment Score"),
            color=alt.value("#4A90E2")
        ).properties(width=900, height=400)
        st.altair_chart(sentiment_line)

        current_price = get_realtime_price_alpha(ticker)
        st.subheader(f"💰 Real-Time Price for **{ticker}**: **${current_price}**")

        st.subheader(f"📉 Stock Price Trend for **{ticker}**")
        price_data = get_historical_price_alpha(ticker)

        if not price_data.empty:
            price_chart = alt.Chart(price_data).mark_line().encode(
                x=alt.X('Date:T', title="Date"),
                y=alt.Y('Close:Q', title="Stock Price ($)"),
                color=alt.value("#FFA500")
            ).properties(width=900, height=400)
            st.altair_chart(price_chart)

            st.subheader("📊 Sentiment and Price Correlation")
            combined_df = pd.merge(data, price_data, left_on="created_utc", right_on="Date", how="inner")
            if not combined_df.empty:
                correlation = combined_df['title_compound'].corr(combined_df['Close'])
                st.metric(label="Correlation (Sentiment vs. Price)", value=f"{correlation:.2f}")
            else:
                st.warning("⚠️ No overlapping data for correlation.")
        else:
            st.warning("⚠️ No historical price data available for this ticker.")

        st.download_button(label="💾 Download Data as CSV", data=data.to_csv(), file_name=f"{ticker}_reddit_posts_cleaned.csv")
    else:
        st.error("❌ Failed to fetch data. Please check your ticker symbol and try again.")

st.markdown("🔗 Created by **Albert Paradell** - Real-time stock analysis powered by Reddit sentiment.")
