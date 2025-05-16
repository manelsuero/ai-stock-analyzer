import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar: Datos de mercado ─────────────────────────────────────────────────
st.sidebar.header("Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

# ── Sidebar: Sentimiento en Reddit ────────────────────────────────────────────
st.sidebar.header("💬 Reddit Sentiment Options")
days  = st.sidebar.slider("Days of history",           3, 30, 14)
size  = st.sidebar.slider("Max comments to analyze",  50, 500, 300)
pages = st.sidebar.slider("Pages of results to fetch", 1,   5,   3)

# ── Inicializa VADER ──────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

# ── Función: Traer comentarios de Pushshift API ───────────────────────────────
def fetch_reddit_comments(ticker, after, before, size, max_pages):
    SUBS = ["stocks", "investing", "wallstreetbets"]
    all_comments = []
    params = {
        "q":         f"{ticker} OR ${ticker}",
        "subreddit": SUBS,
        "after":     after,
        "before":    before,
        "size":      size
    }
    for _ in range(max_pages):
        r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=params)
        batch = r.json().get("data", [])
        if not batch:
            break
        all_comments.extend(batch)
        # desplazamos "before" para la siguiente página
        params["before"] = batch[-1]["created_utc"]
    # filtramos
    filtered = [
        c for c in all_comments
        if len(c.get("body","")) > 20 and "http" not in c.get("body","")
    ]
    if not filtered:
        return pd.DataFrame(columns=["date","body"])
    df = pd.DataFrame({
        "date": pd.to_datetime([c["created_utc"] for c in filtered], unit="s"),
        "body": [c["body"] for c in filtered]
    })
    return df

# ── Función: Calcular sentimiento diario ──────────────────────────────────────
def analyze_sentiment(df_comments):
    if df_comments.empty:
        return pd.Series(dtype=float)
    df_comments["score"] = df_comments["body"].apply(lambda t: sia.polarity_scores(t)["compound"])
    daily = (
        df_comments
        .set_index("date")
        .resample("D")
        .mean()["score"]
        .fillna(method="ffill")
    )
    return daily

# ── Cache para no recargar cada interacción ───────────────────────────────────
@st.cache_data(ttl=3600)
def get_reddit_sentiment(ticker, days, size, pages):
    end_unix   = int(pd.Timestamp.today().timestamp())
    start_unix = int((pd.Timestamp.today() - pd.Timedelta(days=days)).timestamp())
    df_comments = fetch_reddit_comments(ticker, start_unix, end_unix, size, pages)
    # si hay más que 'size', muestreamos
    if len(df_comments) > size:
        df_comments = df_comments.sample(size, random_state=42)
    return analyze_sentiment(df_comments)

# ── Botón principal ──────────────────────────────────────────────────────────
if st.sidebar.button("🔍 Analyze Stock"):

    # ——— 1️⃣ Descarga de datos de Yahoo Finance ————————————————
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data found for “{ticker}”.")
        st.stop()

    # ——— 2️⃣ Cálculo manual de indicadores técnicos —————————————
    # SMA 20
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    # RSI 14
    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + (avg_gain/avg_loss)))
    # MACD & Signal
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ——— 3️⃣ Mostrar gráficos con Matplotlib ————————————————
    st.header("🔍 Technical Indicators")

    # RSI
    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.set_ylabel("RSI")
    ax.legend()
    st.pyplot(fig)

    # SMA20
    st.subheader("SMA20 over Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # MACD
    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["Signal Line"], label="Signal")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.success("✅ Technical indicators loaded. Next: Social Media Sentiment.")

    # ——— 4️⃣ Sentimiento en Reddit —————————————————————————
    sentiment = get_reddit_sentiment(ticker, days, size, pages)
    if sentiment.empty:
        st.warning("No Reddit comments found for that ticker.")
    else:
        st.header("💬 Reddit Sentiment")
        st.line_chart(sentiment)
        st.markdown(f"_Avg daily sentiment over last {days} days ({len(sentiment)} points)._")

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
