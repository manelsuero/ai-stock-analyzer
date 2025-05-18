import streamlit as st
import pandas as pd
import altair as alt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# ─── CONFIGURACIÓN ─────────────────────────────────────────────
st.set_page_config(page_title="📊 News Sentiment Analyzer", layout="wide")
st.title("🗞️ News Sentiment Analyzer")

# ─── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")
ticker = st.sidebar.text_input("Enter Company or Ticker", value="AAPL")
limit = st.sidebar.slider("Number of News Articles", min_value=10, max_value=100, value=50)

# ─── FUNCIONES ────────────────────────────────────────────────
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

# ─── ANÁLISIS ──────────────────────────────────────────────────
if st.sidebar.button("🚀 Analyze"):
    st.info(f"Fetching and analyzing news about **{ticker}**...")
    df = fetch_news_sentiment(ticker, st.secrets["NEWSAPI_KEY"], limit)

    if not df.empty:
        st.success(f"Fetched and analyzed {len(df)} articles for **{ticker}**")

        # MÉTRICAS GENERALES
        avg_compound = df["sentiment_compound"].mean()
        st.metric("🧠 Average Sentiment Score", f"{avg_compound:.2f}")

        # VEREDICTO GENERAL
        pos = df["sentiment_pos"].mean()
        neg = df["sentiment_neg"].mean()
        neu = df["sentiment_neu"].mean()
        if pos > neg and pos > neu:
            verdict = "🟢 Positive"
        elif neg > pos and neg > neu:
            verdict = "🔴 Negative"
        else:
            verdict = "🟡 Neutral"
        st.markdown(f"### Overall Sentiment: {verdict}")

        # DISTRIBUCIÓN DE SENTIMIENTO
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

        # TENDENCIA TEMPORAL
        st.subheader("📈 Sentiment Over Time")
        time_chart = alt.Chart(df).mark_line().encode(
            x=alt.X("published_at:T", title="Date"),
            y=alt.Y("sentiment_compound:Q", title="Compound Sentiment"),
            tooltip=["title", "sentiment_compound"]
        ).properties(width=900, height=400)
        st.altair_chart(time_chart)

        # TABLA DE NOTICIAS
        st.subheader("📰 News Table")
        st.dataframe(df[["published_at", "title", "sentiment_compound", "source", "url"]])

        # DESCARGA CSV
        st.download_button(
            "💾 Download CSV",
            df.to_csv(index=False),
            file_name=f"{ticker}_news_sentiment.csv"
        )
    else:
        st.warning("No data returned from NewsAPI.")
