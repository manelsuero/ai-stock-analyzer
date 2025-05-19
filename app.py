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
st.set_page_config(page_title="\ud83d\udccb AI Stock Analyzer", layout="wide")
st.title("\ud83d\udcc8 AI Stock Analyzer with Technical & News Sentiment")

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("\ud83d\udd0d Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today())
news_limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)
investor_type = st.sidebar.selectbox("Investor Profile", ["Day Trader", "Swing Trader", "Long-Term Investor"])

# ─── API KEYS ──────────────────────────────────────────────────────────
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ─── ANÁLISIS TÉCNICO ──────────────────────────────────────────────────
st.header("1\ufe0f\ufe0f Technical Indicators")
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error(f"No market data for \"{ticker}\" in that range.")
    st.stop()

# Calcular indicadores
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Graficar indicadores
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI')
ax.legend(loc="upper left")
st.pyplot(fig)

st.subheader("SMA20 vs Close Price")
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

st.success("\u2705 Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ─── ANÁLISIS DE NOTICIAS ─────────────────────────────────────────────
st.header("2\ufe0f\ufe0f News Sentiment Analysis")

def fetch_news_sentiment(ticker, api_key, limit=50):
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&pageSize={limit}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        st.error(f"Error from NewsAPI: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()
    results = []

    for article in data.get("articles", []):
        title = article["title"]
        content = article["description"] or ""
        combined_text = f"{title} {content}"
        sentiment = analyzer.polarity_scores(combined_text)

        results.append({
            "title": title,
            "content": content,
            "published_at": article["publishedAt"],
            "source": article["source"]["name"],
            "url": article["url"],
            "sentiment_compound": sentiment["compound"]
        })

    df_news = pd.DataFrame(results)
    df_news["published_at"] = pd.to_datetime(df_news["published_at"])
    return df_news

if NEWSAPI_KEY:
    df_news = fetch_news_sentiment(ticker, NEWSAPI_KEY, news_limit)

    if not df_news.empty:
        avg_compound = df_news["sentiment_compound"].mean()
        st.metric("\ud83e\udde0 Avg Sentiment Score", f"{avg_compound:.2f}")

        # ─── GRAFICO CORRELACION ───────────────────────────────────────
        st.markdown("---")
        st.header("\ud83d\udcc9 Price vs Sentiment Over Time")

        try:
            df_news['date'] = df_news['published_at'].dt.date
            sentiment_daily = df_news.groupby('date')['sentiment_compound'].mean().reset_index()
            sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])

            df_price = df.reset_index()
            df_price['date'] = df_price['Date'].dt.date
            df_price['date'] = pd.to_datetime(df_price['date'])

            merged = pd.merge(df_price, sentiment_daily, on='date', how='inner')

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Stock Price', color='tab:blue')
            ax1.plot(merged['date'], merged['Close'], color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Sentiment', color='tab:orange')
            ax2.plot(merged['date'], merged['sentiment_compound'], color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)

            correlation = merged['Close'].corr(merged['sentiment_compound'])
            st.metric("\ud83d\udcc8 Correlation (Price vs Sentiment)", f"{correlation:.2f}")
        except Exception as e:
            st.warning(f"\u26a0\ufe0f Correlation plot error: {str(e)}")

        # ─── FINAL RECOMMENDATION ─────────────────────────────────────
        st.markdown("---")
        st.header("\ud83d\udccd Final Recommendation")

        try:
            client = OpenAI(api_key=OPENAI_KEY)

            final_prompt = f"""
You are a financial analyst advising a {investor_type}.

Based on the following indicators:
- RSI: {df['RSI'].iloc[-1]:.2f}
- MACD: {df['MACD'].iloc[-1]:.2f}
- SMA20: {df['SMA20'].iloc[-1]:.2f}
- News sentiment score: {avg_compound:.2f}

Give a final decision tailored to this investor type: should they BUY, HOLD, or DON'T BUY the stock?
Start your response directly with: \"BUY\", \"HOLD\", or \"DON'T BUY\", followed by a short reason (max 2 lines).
"""

            decision_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": final_prompt}]
            )

            decision_text = decision_response.choices[0].message.content.strip()

            if decision_text.upper().startswith("BUY"):
                st.success(f"\ud83d\udfe2 **{decision_text}**")
            elif decision_text.upper().startswith("HOLD"):
                st.warning(f"\ud83d\udfe1 **{decision_text}**")
            elif decision_text.upper().startswith("DON'T BUY"):
                st.error(f"\ud83d\udd34 **{decision_text}**")
            else:
                st.info(f"\ud83e\udd16 {decision_text}")

        except Exception as e:
            st.warning(f"\u26a0\ufe0f Couldn't generate final recommendation: {str(e)}")

else:
    st.warning("\ud83d\udd11 Please set your NEWSAPI_KEY in Streamlit Secrets.")
