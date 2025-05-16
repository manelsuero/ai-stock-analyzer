import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar común ──────────────────────────────────────────────────────────────
st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

analyze = st.sidebar.button("🔍 Analyze Stock")

if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Análisis técnico
# ────────────────────────────────────────────────────────────────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No data for “{ticker}”.")
    st.stop()

# SMA 20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
# RSI 14
delta    = df['Close'].diff()
gain     = delta.where(delta > 0, 0)
loss     = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + avg_gain/avg_loss))
# MACD & Signal
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

st.header("🔍 Technical Indicators")

# RSI
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI')
ax.legend()
st.pyplot(fig)

# SMA
st.subheader("SMA 20 over Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# MACD
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'], label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal Line')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.success("✅ Technical indicators loaded. Next: Social Media Sentiment.")

# ────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Social Media Sentiment (Reddit)
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("💬 Reddit Sentiment Options")
days  = st.sidebar.slider("Days of history",           3, 30, 14)
size  = st.sidebar.slider("Max comments/posts to fetch", 50, 500, 300)
pages = st.sidebar.slider("Pages of results to fetch", 1,   5,   3)

sia = SentimentIntensityAnalyzer()

def fetch_reddit_data(ticker, after, before, size, max_pages):
    SUBS = ["stocks", "investing", "wallstreetbets", ticker]
    texts = []

    # Comentarios
    params = {"q":f"{ticker} OR ${ticker}", "subreddit": SUBS,
              "after":after, "before":before, "size":size}
    for _ in range(max_pages):
        r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=params)
        batch = r.json().get("data", [])
        if not batch: break
        texts += [c["body"] for c in batch if len(c["body"])>10]
        params["before"] = batch[-1]["created_utc"]

    # Títulos de posts
    params = {"q":f"{ticker} OR ${ticker}", "subreddit": SUBS,
              "after":after, "before":before, "size":size}
    for _ in range(max_pages):
        r = requests.get("https://api.pushshift.io/reddit/submission/search/", params=params)
        batch = r.json().get("data", [])
        if not batch: break
        texts += [p["title"] for p in batch if len(p["title"])>10]
        params["before"] = batch[-1]["created_utc"]

    if not texts:
        return pd.DataFrame(columns=["date","body"])
    # Distribuimos fechas uniformemente entre después y antes
    ts = np.linspace(after, before, len(texts)).astype(int)
    return pd.DataFrame({
        "date": pd.to_datetime(ts, unit='s'),
        "body": texts
    })

@st.cache_data(ttl=3600)
def get_reddit_sentiment(ticker, days, size, pages):
    now = pd.Timestamp.today()
    end_u   = int(now.timestamp())
    start_u = int((now - pd.Timedelta(days=days)).timestamp())
    df_txts = fetch_reddit_data(ticker, start_u, end_u, size, pages)
    if df_txts.empty:
        return pd.Series(dtype=float)
    df_txts["score"] = df_txts["body"].apply(lambda t: sia.polarity_scores(t)["compound"])
    daily = df_txts.set_index("date").resample("D").mean()["score"].fillna(method="ffill")
    return daily

sentiment = get_reddit_sentiment(ticker, days, size, pages)
if sentiment.empty:
    st.warning("⚠️ No Reddit data found for that ticker in the selected range.")
else:
    st.header("💬 Reddit Sentiment")
    st.line_chart(sentiment)
    st.markdown(f"_Average daily sentiment over the last {days} days ({len(sentiment)} points)._")

# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣ News Analysis (pendiente)
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.info("📰 Next up: News Analysis (coming soon!)")
