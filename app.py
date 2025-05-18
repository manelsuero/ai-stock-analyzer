import streamlit as st
import pandas as pd
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

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# ─── PLOTS ───────────────────────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")

try:
    st.line_chart(df[["Close", "SMA20", "EMA20"]])
    st.line_chart(df[["RSI"]])
    st.line_chart(df[["BB_Upper", "BB_Mid", "BB_Lower"]])
    st.line_chart(df[["MACD", "Signal Line"]])
except KeyError:
    st.warning("⚠️ Not enough data for some indicators.")

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
