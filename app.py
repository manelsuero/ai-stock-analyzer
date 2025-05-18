import streamlit as st
import pandas as pd
import altair as alt
import praw
import re
import os
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────
# ✅ Configuración inicial
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧠 Reddit Stock Sentiment", layout="wide")
st.title("🧠 Reddit Stock Sentiment")

# Descargar recursos NLTK si no están
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
setup_nltk()

# ──────────────────────────────────────────────────────────────
# 🔐 Conexión a Reddit API
# ──────────────────────────────────────────────────────────────
def connect_to_reddit():
    try:
        reddit = praw.Reddit(
            client_id=st.secrets["REDDIT_CLIENT_ID"],
            client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
            user_agent=st.secrets["REDDIT_USER_AGENT"],
            check_for_async=False
        )
        _ = reddit.user.me()  # Verificación real de conexión
        st.success("✅ Connected to Reddit API.")
        return reddit
    except Exception as e:
        st.error(f"❌ Reddit connection failed: {e}")
        st.stop()

# ──────────────────────────────────────────────────────────────
# 🧹 Limpieza y análisis de sentimiento
# ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def score_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# ──────────────────────────────────────────────────────────────
# 🔎 Función principal para buscar posts
# ──────────────────────────────────────────────────────────────
def fetch_reddit_posts(ticker, limit=50):
    reddit = connect_to_reddit()
    posts = []
    subreddit = reddit.subreddit("all")

    for submission in subreddit.search(ticker, limit=limit):
        cleaned_title = clean_text(submission.title)
        sentiment = score_sentiment(cleaned_title)

        posts.append({
            "date": datetime.utcfromtimestamp(submission.created_utc),
            "title": submission.title,
            "score": sentiment["compound"],
            "pos": sentiment["pos"],
            "neg": sentiment["neg"],
            "neu": sentiment["neu"],
            "upvotes": submission.score,
            "comments": submission.num_comments,
            "url": submission.url
        })

    return pd.DataFrame(posts)

# ──────────────────────────────────────────────────────────────
# 🎛️ Interfaz de usuario
# ──────────────────────────────────────────────────────────────
st.sidebar.title("📥 Sentiment Analyzer")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

if st.sidebar.button("🚀 Analyze"):
    st.info(f"Analyzing Reddit sentiment for **{ticker}**...")
    df = fetch_reddit_posts(ticker, limit=limit)

    if not df.empty:
        avg_score = df["score"].mean()
        st.metric("Average Sentiment", f"{avg_score:.2f}")

        sentiment_type = (
            "🟢 Positive" if avg_score > 0.05 else
            "🔴 Negative" if avg_score < -0.05 else
            "🟡 Neutral"
        )
        st.markdown(f"### Overall Sentiment: {sentiment_type}")

        st.subheader("📊 Sentiment Distribution")
        chart_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Score": [df["pos"].mean(), df["neg"].mean(), df["neu"].mean()]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x="Sentiment",
            y="Score",
            color="Sentiment"
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("📈 Sentiment Over Time")
        df_sorted = df.sort_values("date")
        time_chart = alt.Chart(df_sorted).mark_line().encode(
            x="date:T",
            y="score:Q"
        )
        st.altair_chart(time_chart, use_container_width=True)

        st.subheader("📝 Sample Reddit Posts")
        for _, row in df.head(10).iterrows():
            st.markdown(f"""
                **{row['date'].strftime('%Y-%m-%d %H:%M')}** · Sentiment: {row['score']:.2f} · 👍 {row['upvotes']} · 💬 {row['comments']}  
                > {row['title']}  
                [🔗 View on Reddit]({row['url']})
            """)
    else:
        st.warning("No posts found for this ticker.")
