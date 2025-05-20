# 📈 AI Stock Analyzer with Technical & News Sentiment
# FULL VERSION INCLUDING:
# 1. README info
# 2. Indicator explanations under each graph
# 3. Field to insert OpenAI API Key manually (to avoid shared cost)

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

# ─── CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer with Technical & News Sentiment")

# ─── README SECTION ───────────────────────────────────────────────────
st.markdown("""
## 📘 About this App
This app provides:
- 📊 Technical analysis (RSI, SMA20, MACD)
- 📰 News sentiment analysis (using NewsAPI and Vader)
- 🤖 AI-generated insights and recommendations (OpenAI)

You can choose your stock ticker, time range, number of news, and investor type.
""")

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today())
news_limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)
investor_type = st.sidebar.selectbox("Investor Profile", ["Day Trader", "Swing Trader", "Long-Term Investor"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI API Key")
OPENAI_KEY = st.sidebar.text_input("Paste your OpenAI API Key", type="password")

NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")

# ─── TECHNICAL ANALYSIS ───────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error(f"No market data for \"{ticker}\" in that range.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# RSI14
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

# MACD & Signal Line
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# RSI Plot
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI')
ax.legend(loc="upper left")
st.pyplot(fig)
st.caption("**RSI (Relative Strength Index):** Measures the speed and change of recent price movements. RSI values above 70 indicate overbought conditions; below 30 indicate oversold.")

# SMA vs Close
st.subheader("SMA20 vs Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close Price')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.set_ylabel('Price')
ax.legend(loc="upper left")
st.pyplot(fig)
st.caption("**SMA20 (Simple Moving Average 20):** A 20-day average of closing prices. Helps identify trends and support/resistance levels.")

# MACD
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'], label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal Line')
ax.set_ylabel('Value')
ax.legend(loc="upper left")
st.pyplot(fig)
st.caption("**MACD (Moving Average Convergence Divergence):** Highlights trend changes and momentum by comparing short- and long-term EMAs. Crossovers indicate buy/sell signals.")

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ─── ANÁLISIS DE NOTICIAS ─────────────────────────────────────────────
st.header("2️⃣ News Sentiment Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    st.info(f"Fetching and analyzing news about **{ticker}**...")

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
                "sentiment_pos": sentiment["pos"],
                "sentiment_neg": sentiment["neg"],
                "sentiment_neu": sentiment["neu"],
                "sentiment_compound": sentiment["compound"]
            })

        df = pd.DataFrame(results)
        df["published_at"] = pd.to_datetime(df["published_at"])
        return df

    df_news = fetch_news_sentiment(ticker, NEWSAPI_KEY, news_limit)

    if not df_news.empty:
        st.success(f"Fetched and analyzed {len(df_news)} articles for **{ticker}**")

        avg_compound = df_news["sentiment_compound"].mean()
        st.metric("🧐 Average Sentiment Score", f"{avg_compound:.2f}")

        pos = df_news["sentiment_pos"].mean()
        neg = df_news["sentiment_neg"].mean()
        neu = df_news["sentiment_neu"].mean()

        if pos > neg and pos > neu:
            verdict = "🟢 Positive"
        elif neg > pos and neg > neu:
            verdict = "🔴 Negative"
        else:
            verdict = "🟡 Neutral"

        st.markdown(f"### Overall Sentiment: {verdict}")

        st.subheader("📊 Sentiment Distribution")
        sentiment_dist = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Score": [pos, neg, neu]
        })
        chart = alt.Chart(sentiment_dist).mark_bar().encode(
            x=alt.X("Sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="Score",
            color=alt.Color("Sentiment", scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"],
                range=["#4CAF50", "#FFC107", "#F44336"]
            ))
        ).properties(width=700, height=300)
        st.altair_chart(chart)

        st.subheader("📈 Sentiment Over Time")
        time_chart = alt.Chart(df_news).mark_line().encode(
            x=alt.X("published_at:T", title="Date"),
            y=alt.Y("sentiment_compound:Q", title="Compound Sentiment"),
            tooltip=["title", "sentiment_compound"]
        ).properties(width=900, height=400)
        st.altair_chart(time_chart)

        st.subheader("📰 News Table")
        st.dataframe(df_news[["published_at", "title", "sentiment_compound", "source", "url"]])

        st.download_button(
            "📀 Download CSV",
            df_news.to_csv(index=False),
            file_name=f"{ticker}_news_sentiment.csv"
        )

        # ── IA CONCLUSIÓN GPT ───────────────────────────────────────
        st.markdown("---")
        st.header("🤖 AI Stock Insight")

        if OPENAI_KEY:
            prompt = f"""
            Ticker: {ticker}
            RSI: {df['RSI'].iloc[-1]:.2f}
            MACD: {df['MACD'].iloc[-1]:.2f}
            SMA20: {df['SMA20'].iloc[-1]:.2f}
            News Sentiment: {avg_compound:.2f} ({verdict})

            You are a financial analyst. Based on the technical indicators and the news sentiment,
            provide a short and clear analysis of the stock situation in English.
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


# ─── FINAL DECISION INDICATOR ─────────────────────────────────────
st.markdown("---")
st.header("📍 Final Recommendation")

try:
    final_prompt = f"""
    You are a financial analyst advising a {investor_type}.

    Based on the following indicators:
    - RSI: {df['RSI'].iloc[-1]:.2f}
    - MACD: {df['MACD'].iloc[-1]:.2f}
    - SMA20: {df['SMA20'].iloc[-1]:.2f}
    - News sentiment score: {avg_compound:.2f} ({verdict})

    Give a final decision tailored to this investor type: should they BUY, HOLD, or DON'T BUY the stock?
    Start your response directly with: "BUY", "HOLD", or "DON'T BUY", followed by a short reason (max 2 lines).
    """

    decision_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": final_prompt}]
    )

    decision_text = decision_response.choices[0].message.content.strip()

    # Show visual badge
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
