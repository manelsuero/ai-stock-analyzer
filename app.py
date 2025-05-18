import streamlit as st
import pandas as pd
import altair as alt
import nltk
from sentiment_analysis import fetch_reddit_posts
from data_processing import get_realtime_price_alpha, get_historical_price_alpha

# Descargar recursos NLTK necesarios
nltk.download("punkt")
nltk.download("stopwords")

# Configuración de la página
st.set_page_config(page_title="📊 Real-Time Stock Sentiment Dashboard", layout="wide")

# Sidebar de entrada
st.sidebar.title("🔎 Stock Sentiment Analysis")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
limit = st.sidebar.slider("Number of Reddit Posts", 10, 100, 50)

if st.sidebar.button("🚀 Fetch and Analyze"):
    st.info(f"Fetching and analyzing Reddit posts for **{ticker}**...")
    data = fetch_reddit_posts(ticker, limit=limit)

    if data is not None:
        st.success(f"Data for **{ticker}** successfully loaded! ✅")

        # Sentiment Overview
        st.subheader(f"📊 Sentiment Overview for **{ticker}**")
        avg_compound = data["title_compound"].mean()
        sentiment_counts = data[["title_pos", "title_neg", "title_neu"]].mean()

        if sentiment_counts["title_pos"] > sentiment_counts["title_neg"] and sentiment_counts["title_pos"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🟢 Positive"
        elif sentiment_counts["title_neg"] > sentiment_counts["title_pos"] and sentiment_counts["title_neg"] > sentiment_counts["title_neu"]:
            sentiment_verdict = "🔴 Negative"
        else:
            sentiment_verdict = "🟡 Neutral"

        st.metric(label="Average Sentiment Score", value=f"{avg_compound:.2f}")
        st.markdown(f"**Overall Sentiment:** {sentiment_verdict}")

        # Sentiment Distribution
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Ratio": [sentiment_counts["title_pos"], sentiment_counts["title_neg"], sentiment_counts["title_neu"]]
        })
        st.subheader("📊 Sentiment Distribution")
        bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X("Sentiment", sort=["Positive", "Neutral", "Negative"]),
            y="Ratio",
            color=alt.Color("Sentiment", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                         range=["#4CAF50", "#FFC107", "#F44336"]))
        ).properties(width=700, height=300)
        st.altair_chart(bar_chart)

        # Sentiment Trend
        st.subheader("📈 Sentiment Trend Over Time")
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        sentiment_line = alt.Chart(data).mark_line().encode(
            x=alt.X('created_utc:T', title="Date"),
            y=alt.Y('title_compound:Q', title="Sentiment Score"),
            color=alt.value("#4A90E2")
        ).properties(width=900, height=400)
        st.altair_chart(sentiment_line)

        # Real-time price
        current_price = get_realtime_price_alpha(ticker)
        st.subheader(f"💰 Real-Time Price for **{ticker}**: **${current_price}**")

        # Historical price
        st.subheader(f"📉 Stock Price Trend for **{ticker}**")
        price_data = get_historical_price_alpha(ticker)

        if not price_data.empty:
            price_chart = alt.Chart(price_data).mark_line().encode(
                x=alt.X('Date:T', title="Date"),
                y=alt.Y('Close:Q', title="Stock Price ($)"),
                color=alt.value("#FFA500")
            ).properties(width=900, height=400)
            st.altair_chart(price_chart)

            # Correlation
            st.subheader("📊 Sentiment and Price Correlation")
            combined_df = pd.merge(data, price_data, left_on="created_utc", right_on="Date", how="inner")
            if not combined_df.empty:
                correlation = combined_df['title_compound'].corr(combined_df['Close'])
                st.metric(label="Correlation (Sentiment vs. Price)", value=f"{correlation:.2f}")
            else:
                st.warning("⚠️ No overlapping data for correlation.")
        else:
            st.warning("⚠️ No historical price data available for this ticker.")

        st.download_button(label="💾 Download Data as CSV", data=data.to_csv(), file_name=f"{ticker}_reddit_posts_cleaned.csv")
    else:
        st.error("❌ Failed to fetch data. Please check your ticker symbol and try again.")

st.markdown("🔗 Created by **Manel Suero** - Real-time stock analysis powered by Reddit sentiment.")
