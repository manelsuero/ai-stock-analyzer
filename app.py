import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import finnhub

# ── 0. Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

# ── 2. Finnhub client ────────────────────────────────────────────────────────────
fh = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])

# ── 3. Analysis trigger ──────────────────────────────────────────────────────────
if st.sidebar.button("🔍 Analyze Stock"):

    # 3.1️⃣ Fetch price data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No market data found for “{ticker}” – check the symbol.")
        st.stop()

    # ── 4. Technical Analysis ─────────────────────────────────────────────────────
    # 4.1 Compute SMA20
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # 4.2 Compute RSI
    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # 4.3 Compute MACD & signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 4.4 Display charts
    st.header("🔍 Technical Indicators")

    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.set_ylabel("RSI")
    ax.legend()
    st.pyplot(fig)

    st.subheader("SMA 20 over Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close Price")
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["Signal Line"], label="Signal Line")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.success("✅ Technical indicators loaded. Next up: Social-Media Sentiment")

    # ── 5. Social-Media Sentiment via Finnhub ────────────────────────────────────
    @st.cache_data(ttl=3600)
    def get_social_sentiment(ticker, start_iso, end_iso):
        """Returns DataFrame of daily mention_score from Reddit via Finnhub."""
        raw = fh.stock_social_sentiment(
            ticker,
            _from=start_iso,
            to=end_iso
        )
        # pick Reddit channel
        df_soc = pd.DataFrame(raw.get("reddit", []))
        if df_soc.empty:
            return df_soc
        df_soc["mention_score"] = df_soc["mention_score"].astype(float)
        df_soc["timestamp"]     = pd.to_datetime(df_soc["timestamp"], unit="s")
        return df_soc.set_index("timestamp").resample("D").mean()["mention_score"].fillna(method="ffill")

    sentiment = get_social_sentiment(
        ticker,
        start_date.isoformat(),
        end_date.isoformat()
    )

    st.header("💬 Social Sentiment (Reddit)")
    if sentiment.empty:
        st.warning("No social-media sentiment data available for this ticker.")
    else:
        st.line_chart(sentiment)
        st.markdown(
            f"_Daily average “mention_score” from {start_date} to {end_date} (source: Finnhub)._"
        )

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
