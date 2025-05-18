import streamlit as st
import pandas as pd
import praw
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Configuración inicial
st.set_page_config(page_title="Reddit Stock Sentiment", layout="wide")
st.title("🧠 Reddit Stock Sentiment")

# Descargar recursos de NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

# Conexión a Reddit
def connect_to_reddit():
    try:
        reddit = praw.Reddit(
            client_id=st.secrets["REDDIT_CLIENT_ID"],
            client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
            user_agent=st.secrets["REDDIT_USER_AGENT"]
        )
        reddit.read_only = True
        st.success("✅ Connected to Reddit API.")
        return reddit
    except Exception as e:
        st.error(f"❌ Reddit connection failed: {e}")
        return None

# Limpiar texto
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Puntuar sentimiento
def score_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# Extraer posts
def fetch_reddit_posts(ticker, limit=50):
    reddit = connect_to_reddit()
    if reddit is None:
        return None

    posts = []
    try:
        for submission in reddit.subreddit("all").search(ticker, sort="new", limit=limit):
            cleaned_title = clean_text(submission.title)
            title_scores = score_sentiment(cleaned_title)

            posts.append({
                "title": submission.title,
                "score": title_scores["compound"],
                "created_utc": datetime.utcfromtimestamp(submission.created_utc),
                "upvotes": submission.score,
                "comments": submission.num_comments,
                "url": submission.url
            })
        return pd.DataFrame(posts)
    except Exception as e:
        st.error(f"❌ Error fetching Reddit posts: {e}")
        return None

# Botón
if st.sidebar.button("🚀 Analyze"):
    st.info(f"Analyzing Reddit sentiment for **{ticker}**...")
    df = fetch_reddit_posts(ticker, limit=limit)

    if df is not None and not df.empty:
        avg_score = df["score"].mean()
        st.metric("Average Sentiment", f"{avg_score:.3f}")

        st.subheader("📈 Sentiment Over Time")
        df = df.sort_values("created_utc")
        df["created_utc"] = pd.to_datetime(df["created_utc"])

        st.line_chart(df.set_index("created_utc")["score"])

        st.subheader("📄 Latest Reddit Posts")
        for _, row in df.head(5).iterrows():
            st.markdown(f"""
                **{row['created_utc'].strftime('%Y-%m-%d %H:%M')}** | 👍 {row['upvotes']} | 💬 {row['comments']}  
                > {row['title']}  
                [🔗 View Post]({row['url']})
            """)
    else:
        st.warning("No posts found or failed to load.")
