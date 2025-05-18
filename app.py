import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import openai  # 👈 Añadido para IA

# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────
st.set_page_config(page_title="📊 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer with Technical & News Sentiment")

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today())
news_limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)

# ─── ANÁLISIS TÉCNICO ──────────────────────────────────────────────────
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

# SMA vs Close
st.subheader("SMA20 vs Close Price")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close Price')
ax.plot(df.index, df['SMA20'], label='SMA20')
ax.set_ylabel('Price')
ax.legend(loc="upper left")
st.pyplot(fig)

# MACD & Signal
st.subheader("MACD & Signal Line")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACD'], label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal Line')
ax.set_ylabel('Value')
ax.legend(loc="upper left")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ─── ANÁLISIS DE NOTICIAS ─────────────────────────────────────────────
st.header("2️⃣ News Sentiment Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")  # 👈 Clave de OpenAI

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
        st.metric("🧠 Average Sentiment Score", f"{avg_compound:.2f}")

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
            "💾 Download CSV",
            df_news.to_csv(index=False),
            file_name=f"{ticker}_news_sentiment.csv"
        )

        # ── IA CONCLUSIÓN GPT ───────────────────────────────────────
        st.markdown("---")
        st.header("🤖 AI Stock Insight")

        from openai import OpenAI  # ✅ compatible con openai >= 1.0.0

        if st.secrets.get("OPENAI_API_KEY"):
            prompt = f"""
            Ticker: {ticker}
            RSI: {df['RSI'].iloc[-1]:.2f}
            MACD: {df['MACD'].iloc[-1]:.2f}
            SMA20: {df['SMA20'].iloc[-1]:.2f}
            News Sentiment: {avg_compound:.2f} ({verdict})

            Eres un analista financiero. Con base en estos datos técnicos y de sentimiento de noticias,
            genera un resumen claro y breve de la situación actual de esta acción.
            """

            try:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success("🔍 Análisis generado por IA:")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.warning(f"⚠️ Error al generar análisis con OpenAI: {str(e)}")

        else:
            st.warning("🔑 Añade tu OPENAI_API_KEY en los secretos para usar la IA.")

