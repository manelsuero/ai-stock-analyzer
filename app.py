import streamlit as st
import pandas as pd
<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from finnhub import Client as FinnhubClient

# ─── CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ─── SIDEBAR FORM ────────────────────────────────────────────────────────
with st.sidebar.form("options_form"):
    ticker    = st.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL").upper()
    start_dt  = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_dt    = st.date_input("End Date", datetime.today())
    days_news = st.slider("Days of news history", 1, 7, 3)
    max_news  = st.slider("Max articles to fetch", 10, 100, 30)
    analyze   = st.form_submit_button("🔍 Analyze Stock")

if not analyze:
    st.info("👈 Use the sidebar to choose a stock and click Analyze Stock.")
    st.stop()
=======
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
>>>>>>> Stashed changes

# ─── DOWNLOAD MARKET DATA ────────────────────────────────────────────────
df = yf.download(ticker, start=start_dt, end=end_dt)
if df.empty:
    st.error(f"No market data for {ticker} in that range.")
    st.stop()

# ─── INDICATORS ──────────────────────────────────────────────────────────
df["SMA20"] = df["Close"].rolling(20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
delta = df["Close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / roll_down
df["RSI"] = 100 - (100 / (1 + rs))
df["BB_Mid"] = df["Close"].rolling(20).mean()
df["BB_Std"] = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

<<<<<<< Updated upstream
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
=======
        st.subheader(f"📊 Sentiment Overview for **{ticker}**")
        avg_compound = data["title_compound"].mean()
        sentiment_counts = data[["title_pos", "title_neg", "title_neu"]].mean()
>>>>>>> Stashed changes

# ─── PLOTS ───────────────────────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")

try:
    st.line_chart(df[["Close", "SMA20", "EMA20"]])
    st.line_chart(df[["RSI"]])
    st.line_chart(df[["BB_Upper", "BB_Mid", "BB_Lower"]])
    st.line_chart(df[["MACD", "Signal Line"]])
except KeyError:
    st.warning("⚠️ Not enough data for some indicators.")

<<<<<<< Updated upstream
# ─── NEWS FROM FINNHUB ───────────────────────────────────────────────────
st.markdown("---")
st.header("2️⃣ News Analysis")
try:
    fh = FinnhubClient(api_key=st.secrets["FINNHUB_KEY"])
    now_ts  = int(datetime.now().timestamp())
    past_ts = int((datetime.now() - timedelta(days=days_news)).timestamp())
    all_news = fh.general_news("general", min_id=None)
    filtered = [n for n in all_news if past_ts <= n.get("datetime", 0) <= now_ts][:max_news]

    if not filtered:
        st.warning("No news found from Finnhub in selected date range.")
=======
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
>>>>>>> Stashed changes
    else:
        df_news = pd.DataFrame([{
            "Date": datetime.fromtimestamp(n["datetime"]),
            "Headline": n["headline"],
            "Source": n["source"],
            "URL": n["url"]
        } for n in filtered])
        st.dataframe(df_news)
except Exception as e:
    st.error("❌ Error fetching news from Finnhub.")
    st.exception(e)

<<<<<<< Updated upstream
# ─── NEWS FROM NEWSAPI ───────────────────────────────────────────────────
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    st.subheader("🗞️ NewsAPI Sentiment Analysis")
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={max_news}&"
        f"from={(pd.Timestamp.today() - pd.Timedelta(days=days_news)).date()}&"
        f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(news_url, timeout=5).json()
        articles = r.get("articles", [])
        if not articles:
            st.warning("No articles found (API limit or empty results).")
        else:
            analyzer = SentimentIntensityAnalyzer()
            sentiment_data = []
            for a in articles:
                combined_text = f"{a['title']} {a.get('description', '')}"
                sentiment = analyzer.polarity_scores(combined_text)
                sentiment_data.append({
                    "Title": a["title"],
                    "Published": a["publishedAt"],
                    "Sentiment": sentiment["compound"],
                    "Source": a["source"]["name"],
                    "URL": a["url"]
                })
            df_sentiment = pd.DataFrame(sentiment_data)
            st.dataframe(df_sentiment)
    except Exception as e:
        st.error("❌ Error connecting to NewsAPI.")
        st.exception(e)

st.success("✅ Analysis complete. Ready for Social Media module next!")
=======
st.markdown("🔗 Created by **Albert Paradell** - Real-time stock analysis powered by Reddit sentiment.")
>>>>>>> Stashed changes
