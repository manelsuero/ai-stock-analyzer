# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
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
    st.subheader("💬 StockTwits Sentiment Options")
    st_tw_days = st.slider("Days of posts history",1, 14,  7, key="tw_days")
    st_tw_max  = st.slider("Max posts to fetch",   10, 200, 50, key="tw_max")

    analyze = st.form_submit_button("🔍 Analyze Stock")

# Si no has pulsado Analyze, paramos
if not analyze:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Download & fundamental (técnico) indicators ──────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for “{ticker}” in that range.")
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
ax.legend(loc="upper left")
st.pyplot(fig)

st.success("✅ Technical indicators loaded. Next: News Analysis & Sentiment.")
st.markdown("---")

# ── 3. News Analysis via NewsAPI ────────────────────────────────────────
st.header("2️⃣ News Analysis")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
if not NEWSAPI_KEY:
    st.warning("🔑 Please set your NEWSAPI_KEY in Streamlit Secrets.")
else:
    news_url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&pageSize={news_max}&"
        f"from={(pd.Timestamp.today()-pd.Timedelta(days=news_days)).date()}&"
        f"sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
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

st.success("✅ News Analysis loaded. Next: Social Sentiment (StockTwits).")
st.markdown("---")

# ── 4. Social Sentiment (StockTwits + Vader) ────────────────────────────
def fetch_stocktwits(symbol, days, max_posts):
    """
    Función mejorada para obtener datos de StockTwits con mejor manejo de errores
    y procesamiento más robusto de la respuesta.
    """
    import pandas as pd
    import requests
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    # Inicializar el analizador de sentimiento
    sia = SentimentIntensityAnalyzer()
    
    # Calcular timestamps para el filtro de días
    end = int(pd.Timestamp.now().timestamp())
    start = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
    
    # URL base para la API de StockTwits
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    
    try:
        # Realizar la solicitud con manejo de errores mejorado
        response = requests.get(url, timeout=10)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code != 200:
            st.warning(f"Error al obtener datos de StockTwits: {response.status_code}")
            return pd.DataFrame()
            
        # Obtener los datos JSON
        data = response.json()
        
        # Verificar si hay mensajes en la respuesta
        if 'messages' not in data or not data['messages']:
            return pd.DataFrame()
            
        # Limitar la cantidad de mensajes según el parámetro
        msgs = data['messages'][:max_posts]
        
        # Procesar los mensajes
        processed_data = []
        for m in msgs:
            # Convertir la fecha a datetime
            t = pd.to_datetime(m["created_at"])
            
            # Filtrar solo mensajes dentro del rango de fechas especificado
            if pd.Timestamp(start, unit='s') <= t <= pd.Timestamp(end, unit='s'):
                # Obtener el texto del mensaje
                text = m.get("body", "")
                
                # Obtener la categoría de sentimiento si está disponible
                sentiment_data = m.get("entities", {}).get("sentiment", {})
                cat = sentiment_data.get("basic", "Neutral") if sentiment_data else "Neutral"
                
                # Calcular la puntuación de sentimiento utilizando VADER
                comp = sia.polarity_scores(text)["compound"]
                
                # Agregar a los datos procesados
                processed_data.append((t, text, cat, comp))
        
        # Crear DataFrame
        df = pd.DataFrame(processed_data, columns=["date", "text", "cat_sent", "score"])
        
        # Si no hay datos después del filtrado, devolver DataFrame vacío
        if df.empty:
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        st.warning(f"Error al procesar datos de StockTwits: {str(e)}")
        return pd.DataFrame()
