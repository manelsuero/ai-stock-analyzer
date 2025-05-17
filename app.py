import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 0. Configuración inicial ───────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Sidebar: Market + News + Sentiment Options ────────────────────────
st.sidebar.header("🔢 Market & News Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.header("📰 News Options")
news_days = st.sidebar.slider("Days of news history", 1, 7, 3)
news_max  = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

st.sidebar.markdown("---")
st.sidebar.header("💬 StockTwits Sentiment Options")
st_tw_days  = st.sidebar.slider("Days of posts history",    1, 14, 7)
st_tw_max   = st.sidebar.slider("Max posts to fetch",      10, 200, 50)

if st.sidebar.button("🔍 Analyze Stock"):

    # ── 2. Descarga y calcula indicadores técnicos ───────────────────────
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No market data for “{ticker}”.")
        st.stop()

    # SMA20
    df['SMA20'] = df['Close'].rolling(20).mean()
    # RSI14
    delta     = df['Close'].diff()
    gain      = delta.where(delta>0, 0)
    loss      = -delta.where(delta<0, 0)
    avg_gain  = gain.ewm(span=14).mean()
    avg_loss  = loss.ewm(span=14).mean()
    df['RSI'] = 100 - (100/(1 + avg_gain/avg_loss))
    # MACD
    exp12     = df['Close'].ewm(span=12).mean()
    exp26     = df['Close'].ewm(span=26).mean()
    df['MACD']        = exp12 - exp26
    df['Signal Line'] = df['MACD'].ewm(span=9).mean()

    # ── 3. Plot Technical Indicators ────────────────────────────────────
    st.header("1️⃣ Technical Indicators")
    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.legend()
    st.pyplot(fig)

    st.subheader("SMA20 vs Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Close')
    ax.plot(df.index, df['SMA20'], label='SMA20')
    ax.legend()
    st.pyplot(fig)

    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'],        label='MACD')
    ax.plot(df.index, df['Signal Line'], label='Signal')
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")

    # ── 4. News Analysis (ej. NewsAPI) ─────────────────────────────────
    st.header("2️⃣ News Analysis")
    api_key = st.secrets["NEWSAPI_KEY"]  # ya la tienes en Secrets sin [general]
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={news_max}&"
        f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
        f"sortBy=publishedAt&apiKey={api_key}"
    )
    r = requests.get(news_url).json()
    articles = r.get("articles", [])
    if not articles:
        st.warning("No news found (¿API key o límite alcanzado?).")
    else:
        df_news = pd.DataFrame([{
            "datetime": a["publishedAt"],
            "headline": a["title"],
            "source":   a["source"]["name"],
            "url":      a["url"]
        } for a in articles])
        st.dataframe(df_news)

    # ── 5. StockTwits Sentiment Analysis ────────────────────────────────
    st.header("3️⃣ Social Sentiment (StockTwits)")
    sia = SentimentIntensityAnalyzer()

    @st.cache_data(ttl=3600)
    def fetch_stocktwits(symbol, days, max_posts):
        end = int(pd.Timestamp.now().timestamp())
        start = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        msgs = requests.get(url).json().get("messages", [])[:max_posts]
        data = []
        for m in msgs:
            t = pd.to_datetime(m["created_at"])
            body = m["body"]
            # también captura sentimiento categórico si viene en m["entities"]["sentiment"]
            cat = m.get("entities", {}).get("sentiment", {}).get("basic", None)
            comp = sia.polarity_scores(body)["compound"]
            data.append((t, body, cat, comp))
        return pd.DataFrame(data, columns=["date","text","cat_sent","score"])

    df_st = fetch_stocktwits(ticker.upper(), st_tw_days, st_tw_max)
    if df_st.empty:
        st.warning("No StockTwits posts found for that ticker.")
    else:
        # plot evolución diaria del compound score
        daily = df_st.set_index("date")["score"].resample("D").mean().fillna(0)
        st.line_chart(daily)
        st.markdown(f"_Avg compound sentiment over last {st_tw_days} days ({len(daily)} points)_")

        # show breakdown Bull/Bear if categórico
        cat_counts = df_st["cat_sent"].value_counts().reindex(["Bullish","Bearish"], fill_value=0)
        st.bar_chart(cat_counts)

        # opcional: muestra las últimas 5 con más polaridad extrema
        st.subheader("Top 5 posts (most extreme sentiment)")
        extremes = df_st.reindex(df_st["score"].abs().sort_values(ascending=False).index).head(5)
        st.write(extremes[["date","text","cat_sent","score"]])

else:
    st.info("👈 Enter ticker & hit **Analyze Stock** to start.")
