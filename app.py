import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 0. Configuración inicial ────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Sidebar: Market + News + Sentiment Options ────────────────────────
with st.sidebar.form("options"):
    st.header("🔢 Market & News Options")
    st.subheader("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

    st.markdown("---")
    st.subheader("📰 News Options")
    news_days = st.slider("Days of news history",  1, 7,   3, key="news_days")
    news_max  = st.slider("Max articles to fetch",10, 100, 30, key="news_max")

    st.markdown("---")
    st.subheader("💬 Reddit Sentiment Options")
    reddit_days = st.slider("Days of posts history", 1, 14, 7, key="reddit_days")
    reddit_max  = st.slider("Max posts to fetch", 10, 200, 50, key="reddit_max")
    subreddits  = st.text_input("Subreddits to search (comma separated)", 
                               value="stocks,investing,wallstreetbets")

    analyze = st.form_submit_button("🔍 Analyze Stock")

# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Download & fundamental (técnico) indicators ──────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for \"{ticker}\" in that range.")
    st.stop()

# SMA20
df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# RSI14
delta     = df['Close'].diff()
gain      = delta.where(delta > 0, 0)
loss      = -delta.where(delta < 0, 0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

# MACD & Signal Line
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

st.header("1️⃣ Technical Indicators")

# RSI Plot
st.subheader("RSI (14 days)")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label='RSI')
ax.set_ylabel('RSI')
ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)  # Overbought line
ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)  # Oversold line
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
ax.plot(df.index, df['MACD'],        label='MACD')
ax.plot(df.index, df['Signal Line'], label='Signal Line')
ax.set_ylabel('Value')
ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
ax.legend(loc="upper left")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ── 3. News Analysis via NewsAPI ────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
df_news = None

if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={news_max}&"
        f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
        f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(news_url, timeout=5).json()
        articles = r.get("articles", [])
        if not articles:
            st.warning("No news found (API limit or bad key).")
        else:
            df_news = pd.DataFrame([{
                "datetime": a["publishedAt"],
                "headline": a["title"],
                "source":   a["source"]["name"],
                "url":      a["url"]
            } for a in articles])
            df_news["datetime"] = pd.to_datetime(df_news["datetime"])
            st.dataframe(df_news, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")

st.success("✅ News Analysis loaded. Next: Social Sentiment (Reddit).")
st.markdown("---")

# ── 4. Social Sentiment (Reddit + Vader) ────────────────────────────────
st.header("3️⃣ Social Media Sentiment")

# Función para obtener datos de Reddit
def fetch_reddit_sentiment(ticker, subreddits, days, max_posts):
    """
    Obtiene y analiza posts de Reddit relacionados con un ticker específico.
    
    Args:
        ticker (str): Símbolo de la acción
        subreddits (str): Subreddits separados por comas
        days (int): Número de días para buscar atrás
        max_posts (int): Número máximo de posts a recuperar
        
    Returns:
        DataFrame: DataFrame con los posts y su análisis de sentimiento
    """
    # Inicializar analizador de sentimiento
    sia = SentimentIntensityAnalyzer()
    
    # Preparar lista de subreddits
    subreddit_list = [s.strip() for s in subreddits.split(',')]
    subreddit_param = '+'.join(subreddit_list)
    
    # Calcular timestamp para filtro de días
    after_date = datetime.now() - timedelta(days=days)
    after_timestamp = int(after_date.timestamp())
    
    # URL para la API de Pushshift (Reddit)
    url = "https://api.pushshift.io/reddit/search/submission"
    
    # Lista para almacenar los resultados
    all_posts = []
    
    try:
        # Hacer la solicitud para cada subreddit para mejorar resultados
        params = {
            'q': ticker,
            'subreddit': subreddit_param,
            'after': after_timestamp,
            'sort': 'desc',
            'sort_type': 'created_utc',
            'size': max_posts
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            st.warning(f"Error al obtener datos de Reddit: {response.status_code}")
            return pd.DataFrame()
            
        data = response.json()
        
        if 'data' not in data:
            posts = data.get('data', [])
        else:
            posts = data['data']
            
        # Procesar los posts
        for post in posts:
            created_date = datetime.fromtimestamp(post['created_utc'])
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            
            # Combinar título y texto para análisis de sentimiento
            full_text = f"{title} {selftext}"
            
            # Calcular sentimiento
            sentiment = sia.polarity_scores(full_text)
            compound_score = sentiment['compound']
            
            # Determinar categoría de sentimiento
            if compound_score >= 0.05:
                category = "Bullish"
            elif compound_score <= -0.05:
                category = "Bearish"
            else:
                category = "Neutral"
                
            # Añadir a la lista de posts
            all_posts.append({
                'date': created_date,
                'title': title,
                'text': selftext[:200] + '...' if len(selftext) > 200 else selftext,
                'subreddit': post.get('subreddit', ''),
                'score': post.get('score', 0),  # Puntuación del post (upvotes)
                'url': f"https://reddit.com{post.get('permalink', '')}",
                'sentiment_score': compound_score,
                'cat_sent': category
            })
            
        # Crear DataFrame
        df = pd.DataFrame(all_posts)
        
        # Si no hay datos, devolver DataFrame vacío
        if df.empty:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        st.warning(f"Error al procesar datos de Reddit: {str(e)}")
        return pd.DataFrame()

# Variables para el análisis de sentimiento
df_sentiment = None
with st.spinner("Fetching Reddit data..."):
    try:
        df_sentiment = fetch_reddit_sentiment(ticker, subreddits, reddit_days, reddit_max)
    except Exception as e:
        st.error(f"Error getting Reddit data: {str(e)}")

# Mostrar los datos de sentimiento
if df_sentiment is not None and not df_sentiment.empty:
    # Mostrar un resumen de los datos
    st.subheader("Reddit Sentiment Summary")
    
    # Crear columnas para métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = df_sentiment['sentiment_score'].mean()
        st.metric(
            label="Average Sentiment Score", 
            value=f"{avg_score:.3f}",
            delta=None
        )
    
    with col2:
        positive_posts = (df_sentiment['sentiment_score'] > 0.05).sum()
        st.metric(
            label="Positive Posts", 
            value=positive_posts,
            delta=None
        )
    
    with col3:
        negative_posts = (df_sentiment['sentiment_score'] < -0.05).sum()
        st.metric(
            label="Negative Posts", 
            value=negative_posts,
            delta=None
        )
    
    # Gráfico de dispersión de sentimiento
    st.subheader("Sentiment Score Distribution")
    fig, ax = plt.subplots()
    ax.scatter(df_sentiment['date'], df_sentiment['sentiment_score'], alpha=0.6)
    ax.set_ylabel('Sentiment Score')
    ax.set_xlabel('Date')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    st.pyplot(fig)
    
    # Mostrar los subreddits más activos
    if len(df_sentiment) > 0:
        st.subheader("Most Active Subreddits")
        subreddit_counts = df_sentiment['subreddit'].value_counts().head(5)
        fig, ax = plt.subplots()
        subreddit_counts.plot(kind='bar', ax=ax)
        ax.set_ylabel('Number of posts')
        ax.set_xlabel('Subreddit')
        ax.set_title('Post Count by Subreddit')
        st.pyplot(fig)
    
    # Mostrar algunos posts recientes
    st.subheader("Recent Posts")
    recent_posts = df_sentiment.sort_values('date', ascending=False).head(5)
    for _, row in recent_posts.iterrows():
        sentiment_color = "green" if row['sentiment_score'] > 0.05 else "red" if row['sentiment_score'] < -0.05 else "gray"
        st.markdown(f"""
        <div style='border-left: 3px solid {sentiment_color}; padding-left: 10px;'>
            <p style='font-size: 0.8em; color: gray;'>{row['date']} - r/{row['subreddit']}</p>
            <p><strong>{row['title']}</strong></p>
            <p>{row['text']}</p>
            <p style='font-size: 0.9em;'>Sentiment: {row['cat_sent']} | Score: {row['sentiment_score']:.3f}</p>
            <a href="{row['url']}" target="_blank">View on Reddit</a>
        </div>
        <hr>
        """, unsafe_allow_html=True)
else:
    st.warning("No Reddit data available for this ticker or time period.")

st.success("✅ Social Sentiment Analysis loaded.")
st.markdown("---")

st.caption("Desarrollado por AI Stock Analyzer Team | Última actualización: Mayo 2025")
