import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ── 0. Configuración de la página ─────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar común ─────────────────────────────────────────────────────────────
st.sidebar.header("Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

if not ticker:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 1️⃣ Indicadores Técnicos ───────────────────────────────────────────────────
if st.sidebar.button("🔍 Analyze Stock"):
    # Descarga de datos
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data for “{ticker.upper()}”.")
        st.stop()

    # Cálculo SMA20
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    # RSI 14
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain/avg_loss)))
    # MACD & Signal Line
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Visualización
    st.header("1️⃣ Technical Indicators")

    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.set_ylabel('RSI')
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.subheader("SMA 20 over Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.plot(df.index, df['SMA20'], label='SMA20')
    ax.set_ylabel('Price')
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD')
    ax.plot(df.index, df['Signal Line'], label='Signal Line')
    ax.set_ylabel('Value')
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.markdown("---")
    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # ── 2️⃣ News Analysis ────────────────────────────────────────────────────────
    st.header("2️⃣ News Analysis")
    days_news = st.sidebar.slider("Days of news history", 1, 7, 3, key="news_days")
    max_news  = st.sidebar.slider("Max articles to fetch", 10, 100, 30, key="news_max")

    NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
    if not NEWSAPI_KEY:
        st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
    else:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker.upper()}&"
            f"from={(pd.Timestamp.today() - pd.Timedelta(days=days_news)).date()}&"
            f"pageSize={max_news}&"
            f"apiKey={NEWSAPI_KEY}"
        )
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json().get("articles", [])
        except Exception:
            data = []

        if not data:
            st.warning("No news found for that ticker (or API issue).")
        else:
            df_news = pd.DataFrame([{
                "datetime": a["publishedAt"],
                "headline": a["title"],
                "source": a["source"]["name"],
                "url":      a["url"]
            } for a in data])
            df_news["datetime"] = pd.to_datetime(df_news["datetime"])
            st.dataframe(df_news, use_container_width=True)

    st.markdown("---")
    st.success("✅ News Analysis loaded. Next: Social Sentiment (StockTwits).")

    # ── 3️⃣ Social Sentiment (StockTwits) ──────────────────────────────────────
    st.header("3️⃣ Social Sentiment (StockTwits)")
    days_tw = st.sidebar.slider("Days of posts history", 1, 14, 7, key="tw_days")
    max_tw  = st.sidebar.slider("Max posts to fetch",    10, 200, 50, key="tw_max")

    @st.cache_data(ttl=3600)
    def fetch_stocktwits(tkr: str, days: int, max_posts: int) -> pd.DataFrame:
        end = int(pd.Timestamp.now().timestamp())
        start = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
        url = (
            f"https://api.stocktwits.com/api/2/streams/symbol/{tkr}.json"
            f"?limit={max_posts}&since={start}"
        )
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return pd.DataFrame()
            payload = resp.json()
        except Exception:
            return pd.DataFrame()

        msgs = payload.get("messages", [])[:max_posts]
        if not msgs:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date": pd.to_datetime(m["created_at"]),
            "body": m.get("body","")
        } for m in msgs if m.get("body")])
        return df

    df_tw = fetch_stocktwits(ticker.upper(), days_tw, max_tw)
    if df_tw.empty:
        st.warning("No StockTwits messages found for that ticker (or API limit).")
    else:
        # Ejemplo: gráfico de conteo diario de posts
        counts = df_tw.set_index("date").resample("D").size()
        st.bar_chart(counts)
        st.markdown(f"_Total posts fetched: {len(df_tw)} over last {days_tw} days._")

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
