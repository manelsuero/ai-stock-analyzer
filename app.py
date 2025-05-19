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
from openai import OpenAI  # ✅ Nueva forma compatible con openai >= 1.0.0

# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────
st.set_page_config(page_title="📊 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer with Technical, News & Fundamental Analysis")
st.title("📈 AI Stock Analyzer with Technical & News Sentiment")

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Search Parameters")

@@ -24,11 +24,7 @@ investor_type = st.sidebar.selectbox(
    ["Day Trader", "Swing Trader", "Long-Term Investor"]
)

# ─── API KEYS ──────────────────────────────────────────────────────────
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ─── TECHNICAL ANALYSIS ───────────────────────────────────────────────
# ─── ANÁLISIS TÉCNICO ──────────────────────────────────────────────────
st.header("1️⃣ Technical Indicators")
df = yf.download(ticker, start=start_date, end=end_date)


@@ -36,177 +32,214 @@
    st.error(f"No market data for \"{ticker}\" in that range.")
    st.stop()

# Calculate indicators
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
df['STD'] = df['Close'].rolling(20).std()

# Visual tabs
tabs = st.tabs(["📈 Price & SMA", "📊 RSI", "📉 MACD", "🌩️ Volatility"])

with tabs[0]:
    st.subheader("Price and SMA20")
    if 'SMA20' in df.columns:
        st.line_chart(df[['Close', 'SMA20']])
    else:
        st.warning("SMA20 could not be calculated due to insufficient data.")

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
# ─── ANÁLISIS DE NOTICIAS ─────────────────────────────────────────────
st.header("2️⃣ News Sentiment Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ─── NEWS ANALYSIS ─────────────────────────────────────────────────────
st.markdown("---")
st.header("3️⃣ News Sentiment Analysis")
if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    st.info(f"Fetching and analyzing news about **{ticker}**...")

if NEWSAPI_KEY:
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
            combined = f"{title} {content}"
            sentiment = analyzer.polarity_scores(combined)
            combined_text = f"{title} {content}"
            sentiment = analyzer.polarity_scores(combined_text)

            results.append({
                "title": title,
                "sentiment": sentiment["compound"],
                "published_at": article["publishedAt"]
                "content": content,
                "published_at": article["publishedAt"],
                "source": article["source"]["name"],
                "url": article["url"],
                "sentiment_pos": sentiment["pos"],
                "sentiment_neg": sentiment["neg"],
                "sentiment_neu": sentiment["neu"],
                "sentiment_compound": sentiment["compound"]
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
    avg_compound = 0

# ─── AI ANALYSIS ───────────────────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Stock Insight")
        df = pd.DataFrame(results)
        df["published_at"] = pd.to_datetime(df["published_at"])
        return df

if OPENAI_KEY:
    prompt = f"""
    You are an expert financial analyst advising a {investor_type}.
    Based on the following:
    - Ticker: {ticker}
    - RSI: {df['RSI'].iloc[-1]:.2f}
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
    df_news = fetch_news_sentiment(ticker, NEWSAPI_KEY, news_limit)

    try:
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
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
        st.success("🔍 AI-generated Analysis:")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ Error generating analysis with OpenAI: {str(e)}")
else:
    st.warning("🔑 Please set your OPENAI_API_KEY in Streamlit Secrets.")

# ─── FINAL RECOMMENDATION ─────────────────────────────────────────────
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
