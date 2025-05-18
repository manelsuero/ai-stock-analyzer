import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import yfinance as yf
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 0. Configuración inicial ────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer with News Sentiment", layout="wide")
st.title("📈 AI Stock Analyzer with News Sentiment")

# ── 1. Sidebar: Opciones de mercado, noticias y sentimiento ──────────────
with st.sidebar.form("options"):
    st.header("🔢 Market & News Options")
    
    st.subheader("Market Data Options")
    ticker = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date = st.date_input("End Date", value=pd.Timestamp.today())

    st.markdown("---")
    st.subheader("📰 News Options")
    news_days = st.slider("Days of news history", 1, 7, 3, key="news_days")
    news_max = st.slider("Max articles to fetch", 10, 100, 30, key="news_max")

    st.markdown("---")
    st.subheader("💬 Reddit Sentiment Options")
    reddit_days = st.slider("Days of posts history", 1, 14, 7, key="reddit_days")
    reddit_max = st.slider("Max posts to fetch", 10, 200, 50, key="reddit_max")
    subreddits = st.text_input("Subreddits to search (comma separated)", 
                              value="stocks,investing,wallstreetbets")

    analyze = st.form_submit_button("🔍 Analyze Stock")

# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Descarga de datos del mercado y cálculo de indicadores técnicos ──────
st.header("1️⃣ Market Data & Technical Indicators")

with st.spinner("Downloading market data..."):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No market data for \"{ticker}\" in that range.")
        st.stop()

    # Mostrar gráfico del precio
    st.subheader(f"{ticker} Stock Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.set_ylabel('Price')
    ax.set_title(f"{ticker} Stock Price")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # Calcular indicadores técnicos
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
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.set_ylabel('RSI')
    ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)  # Overbought line
    ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)  # Oversold line
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # SMA vs Close
    st.subheader("SMA20 vs Close Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.plot(df.index, df['SMA20'], label='SMA20')
    ax.set_ylabel('Price')
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # MACD & Signal
    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df['MACD'], label='MACD')
    ax.plot(df.index, df['Signal Line'], label='Signal Line')
    ax.set_ylabel('Value')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # Resumen de los indicadores técnicos
    st.subheader("Technical Analysis Summary")
    
    # Último valor de RSI
    last_rsi = df['RSI'].iloc[-1]
    rsi_status = "Overbought 📈" if last_rsi > 70 else "Oversold 📉" if last_rsi < 30 else "Neutral ⚖️"
    
    # SMA vs Close
    sma_status = "Bullish 🟢" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "Bearish 🔴"
    
    # MACD vs Signal Line
    macd_signal = "Bullish 🟢" if df['MACD'].iloc[-1] > df['Signal Line'].iloc[-1] else "Bearish 🔴"
    
    # Crear tabla de resumen
    summary_data = {
        "Indicator": ["RSI (14)", "SMA20", "MACD"],
        "Value": [f"{last_rsi:.2f}", f"{df['SMA20'].iloc[-1]:.2f}", f"{df['MACD'].iloc[-1]:.4f}"],
        "Signal": [rsi_status, sma_status, macd_signal]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ── 3. Análisis de Noticias y Sentimiento ────────────────────────────────
st.header("2️⃣ News Sentiment Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")

if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    with st.spinner(f"Fetching and analyzing news about {ticker}..."):
        # Función para obtener noticias y analizar sentimiento
        def fetch_news_sentiment(ticker, api_key, days=3, limit=30):
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={ticker}&language=en&pageSize={limit}&"
                f"from={(pd.Timestamp.today()-pd.Timedelta(days=days)).date()}&"
                f"sortBy=publishedAt&apiKey={api_key}"
            )
            
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if data.get("status") != "ok":
                    st.error(f"Error from NewsAPI: {data.get('message', 'Unknown error')}")
                    return pd.DataFrame()
                
                analyzer = SentimentIntensityAnalyzer()
                results = []
                
                for article in data.get("articles", []):
                    title = article["title"]
                    content = article.get("description", "") or ""
                    combined_text = f"{title} {content}"
                    sentiment = analyzer.polarity_scores(combined_text)
                    
                    results.append({
                        "datetime": article["publishedAt"],
                        "title": title,
                        "content": content,
                        "source": article["source"]["name"],
                        "url": article["url"],
                        "sentiment_pos": sentiment["pos"],
                        "sentiment_neg": sentiment["neg"],
                        "sentiment_neu": sentiment["neu"],
                        "sentiment_compound": sentiment["compound"]
                    })
                
                df = pd.DataFrame(results)
                if not df.empty:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
                return pd.DataFrame()
        
        # Obtener noticias y analizar sentimiento
        df_news = fetch_news_sentiment(ticker, NEWSAPI_KEY, news_days, news_max)
        
        if df_news.empty:
            st.warning("No news found (API limit or bad key).")
        else:
            st.success(f"✅ Fetched and analyzed {len(df_news)} articles for **{ticker}**")
            
            # MÉTRICAS GENERALES
            avg_compound = df_news["sentiment_compound"].mean()
            st.metric("🧠 Average Sentiment Score", f"{avg_compound:.2f}")
            
            # VEREDICTO GENERAL
            pos = df_news["sentiment_pos"].mean()
            neg = df_news["sentiment_neg"].mean()
            neu = df_news["sentiment_neu"].mean()
            if pos > neg and pos > neu:
                verdict = "🟢 Positive"
            elif neg > pos and neg > neu:
                verdict = "🔴 Negative"
            else:
                verdict = "🟡 Neutral"
            st.markdown(f"### Overall News Sentiment: {verdict}")
            
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
                    range=["#4CAF50", "#F44336", "#FFC107"]
                ))
            ).properties(width=700, height=300)
            st.altair_chart(chart)
            
            # TENDENCIA TEMPORAL
            st.subheader("📈 Sentiment Over Time")
            df_news_sorted = df_news.sort_values("datetime")
            time_chart = alt.Chart(df_news_sorted).mark_line().encode(
                x=alt.X("datetime:T", title="Date"),
                y=alt.Y("sentiment_compound:Q", title="Compound Sentiment"),
                tooltip=["title", "sentiment_compound", "source"]
            ).properties(width=900, height=400)
            st.altair_chart(time_chart)
            
            # TABLA DE NOTICIAS
            st.subheader("📰 News Articles")
            # Aplicar formato condicional basado en sentimiento
            def color_sentiment(val):
                if val > 0.2:
                    return 'background-color: #d4f7dc'
                elif val < -0.2:
                    return 'background-color: #fad3d3'
                else:
                    return 'background-color: #fafafa'
                
            styled_news = df_news[["datetime", "title", "sentiment_compound", "source", "url"]].copy()
            styled_news.columns = ["Date", "Title", "Sentiment", "Source", "URL"]
            styled_news = styled_news.sort_values("Date", ascending=False)
            st.dataframe(
                styled_news.style.applymap(
                    color_sentiment, subset=["Sentiment"]
                ),
                use_container_width=True
            )
            
            # DESCARGA CSV
            st.download_button(
                "💾 Download News Data CSV",
                df_news.to_csv(index=False),
                file_name=f"{ticker}_news_sentiment.csv"
            )

# ── 4. Correlación entre precio y sentimiento (si ambos están disponibles) ───
if 'df' in locals() and 'df_news' in locals() and not df.empty and not df_news.empty:
    st.header("3️⃣ Price & Sentiment Correlation")
    st.subheader("Stock Price vs News Sentiment")
    
    # Agrupar noticias por día y calcular el sentimiento promedio
    df_news['date'] = df_news['datetime'].dt.date
    daily_sentiment = df_news.groupby('date')['sentiment_compound'].mean().reset_index()
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # Preparar el dataframe de precios con fecha formateada
    df_price = df.reset_index()
    df_price['date'] = df_price['Date'].dt.date
    df_price['date'] = pd.to_datetime(df_price['date'])
    
    # Crear el gráfico combinado
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Precio en el eje izquierdo
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color='tab:blue')
    ax1.plot(df_price['date'], df_price['Close'], color='tab:blue', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Sentimiento en el eje derecho
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment Score', color='tab:red')
    ax2.plot(daily_sentiment['date'], daily_sentiment['sentiment_compound'], color='tab:red', label='News Sentiment')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Ajustar el rango del eje de sentimiento para mejor visualización
    ax2.set_ylim(-1, 1)
    
    # Añadir línea de sentimiento neutral
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Configurar leyenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f"{ticker} Price vs News Sentiment")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calcular correlación (solo si hay suficientes días con ambos datos)
    merged_data = pd.merge(df_price, daily_sentiment, on='date', how='inner')
    if len(merged_data) > 3:  # Asegurar que hay suficientes puntos para una correlación significativa
        correlation = merged_data['Close'].corr(merged_data['sentiment_compound'])
        st.metric("Correlación Precio-Sentimiento", f"{correlation:.3f}")
        
        if abs(correlation) > 0.5:
            st.info("Hay una correlación significativa entre el precio y el sentimiento de las noticias.")
        else:
            st.info("No se observa una correlación fuerte entre el precio y el sentimiento de las noticias.")

st.markdown("---")
st.caption("Developed with ❤️ by Manel Suero")
