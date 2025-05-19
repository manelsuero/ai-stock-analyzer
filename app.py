import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from openai import OpenAI

# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────
st.set_page_config(page_title="📊 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer with Technical, News & Fundamental Analysis")

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today())
news_limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)
investor_type = st.sidebar.selectbox("Investor Profile", ["Day Trader", "Swing Trader", "Long-Term Investor"])

# ─── API KEYS ──────────────────────────────────────────────────────────
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ─── TECHNICAL ANALYSIS ───────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error(f"No market data for \"{ticker}\" in that range.")
    st.stop()

# Indicators
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['STD'] = df['Close'].rolling(20).std()  # Volatility

# Plot Indicators
tabs = st.tabs(["📈 Price & SMA", "📊 RSI", "📉 MACD", "🌩️ Volatility"])

with tabs[0]:
    st.subheader("Price and SMA20")
    st.line_chart(df[['Close', 'SMA20']])

with tabs[1]:
    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.axhline(70, color='red', linestyle='--', alpha=0.3)
    ax.axhline(30, color='green', linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with tabs[2]:
    st.subheader("MACD & Signal")
    st.line_chart(df[['MACD', 'Signal Line']])

with tabs[3]:
    st.subheader("Volatility (20-day STD)")
    st.line_chart(df[['STD']])

# ─── FUNDAMENTAL DATA ──────────────────────────────────────────────────
st.markdown("---")
st.header("2️⃣ Company Fundamentals")

try:
    info = yf.Ticker(ticker).info
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", 0)
    pe_ratio = info.get("trailingPE", "N/A")

    st.markdown(f"**Sector:** {sector}")
    st.markdown(f"**Market Cap:** ${market_cap:,}")
    st.markdown(f"**P/E Ratio:** {pe_ratio}")
except Exception as e:
    st.warning("⚠️ Could not load fundamental data.")
    sector = "N/A"
    market_cap = 0
    pe_ratio = "N/A"

# ─── NEWS ANALYSIS ─────────────────────────────────────────────────────
st.markdown("---")
st.header("3️⃣ News Sentiment Analysis")

if NEWSAPI_KEY:
    def fetch_news_sentiment(ticker, api_key, limit=50):
        url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&pageSize={limit}&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()

        if data.get("status") != "ok":
            return pd.DataFrame()

        analyzer = SentimentIntensityAnalyzer()
        results = []
        for article in data.get("articles", []):
            title = article["title"]
            content = article["description"] or ""
            combined = f"{title} {content}"
            sentiment = analyzer.polarity_scores(combined)
            results.append({
                "title": title,
                "sentiment": sentiment["compound"],
                "published_at": article["publishedAt"]
            })
        df_news = pd.DataFrame(results)
        df_news["published_at"] = pd.to_datetime(df_news["published_at"])
        return df_news

    df_news = fetch_news_sentiment(ticker, NEWSAPI_KEY, news_limit)
    if not df_news.empty:
        avg_compound = df_news["sentiment"].mean()
        st.metric("🧠 Average Sentiment Score", f"{avg_compound:.2f}")
        st.line_chart(df_news.set_index("published_at")["sentiment"])
    else:
        avg_compound = 0
        st.warning("No news data retrieved.")
else:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")

# ─── AI ANALYSIS ───────────────────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Stock Insight")

if OPENAI_KEY:
    prompt = f"""
    You are an expert financial analyst advising a {investor_type}.
    Based on the following:
    - Ticker: {ticker}
    - RSI: {df['RSI'].iloc[-1]:.2f} (Check trend direction if possible)
    - MACD: {df['MACD'].iloc[-1]:.2f} vs Signal: {df['Signal Line'].iloc[-1]:.2f}
    - SMA20: {df['SMA20'].iloc[-1]:.2f}
    - Volatility: {df['STD'].iloc[-1]:.2f}
    - Sector: {sector}
    - Market Cap: {market_cap}
    - P/E Ratio: {pe_ratio}
    - News sentiment score: {avg_compound:.2f}

    Give a short summary of the current situation of the stock, noting if technical indicators or fundamentals
    suggest trends or reversals. Then give a recommendation tailored to a {investor_type}.
    """

    try:
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.success("🔍 AI-generated Analysis:")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ Error generating analysis with OpenAI: {str(e)}")
else:
    st.warning("🔑 Please set your OPENAI_API_KEY in Streamlit Secrets.")

# ─── FINAL RECOMMENDATION ─────────────────────────────────────────────
st.markdown("---")
st.header("📍 Final Recommendation")

try:
    final_prompt = f"""
    Investor profile: {investor_type}
    Ticker: {ticker}
    RSI: {df['RSI'].iloc[-1]:.2f}, MACD: {df['MACD'].iloc[-1]:.2f}, SMA20: {df['SMA20'].iloc[-1]:.2f},
    Volatility: {df['STD'].iloc[-1]:.2f}, News Sentiment: {avg_compound:.2f}, Sector: {sector}, P/E: {pe_ratio}

    Based on all this data, provide a final decision for the investor: should they BUY, HOLD or DON'T BUY this stock?
    Start your answer with: BUY, HOLD, or DON'T BUY, followed by 1–2 lines of reasoning.
    """
    decision = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": final_prompt}]
    )
    decision_text = decision.choices[0].message.content.strip()

    if decision_text.upper().startswith("BUY"):
        st.success(f"🟢 **{decision_text}**")
    elif decision_text.upper().startswith("HOLD"):
        st.warning(f"🟡 **{decision_text}**")
    elif decision_text.upper().startswith("DON'T BUY"):
        st.error(f"🔴 **{decision_text}**")
    else:
        st.info(f"🤖 {decision_text}")
except Exception as e:
    st.warning(f"⚠️ Couldn't generate final recommendation: {str(e)}")
