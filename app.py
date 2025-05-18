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

st.success("✅ News Analysis loaded. Next: Social Sentiment (StockTwits).")
st.markdown("---")

# ── 4. Social Sentiment (StockTwits + Vader) ────────────────────────────
st.header("3️⃣ Social Media Sentiment")

# Función para obtener datos de StockTwits
def fetch_stocktwits(symbol, days, max_posts):
    """
    Función para obtener datos de StockTwits con mejor manejo de errores
    y procesamiento más robusto de la respuesta.
    """
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

# Variables para el análisis de sentimiento
df_sentiment = None
with st.spinner("Fetching StockTwits data..."):
    try:
        df_sentiment = fetch_stocktwits(ticker, st_tw_days, st_tw_max)
    except Exception as e:
        st.error(f"Error getting StockTwits data: {str(e)}")

# Mostrar los datos de sentimiento
if df_sentiment is not None and not df_sentiment.empty:
    # Mostrar un resumen de los datos
    st.subheader("StockTwits Sentiment Summary")
    
    # Crear columnas para métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = df_sentiment['score'].mean()
        st.metric(
            label="Average Sentiment Score", 
            value=f"{avg_score:.3f}",
            delta=None
        )
    
    with col2:
        positive_posts = (df_sentiment['score'] > 0.05).sum()
        st.metric(
            label="Positive Posts", 
            value=positive_posts,
            delta=None
        )
    
    with col3:
        negative_posts = (df_sentiment['score'] < -0.05).sum()
        st.metric(
            label="Negative Posts", 
            value=negative_posts,
            delta=None
        )
    
    # Gráfico de dispersión de sentimiento
    st.subheader("Sentiment Score Distribution")
    fig, ax = plt.subplots()
    ax.scatter(df_sentiment['date'], df_sentiment['score'], alpha=0.6)
    ax.set_ylabel('Sentiment Score')
    ax.set_xlabel('Date')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    st.pyplot(fig)
    
    # Mostrar algunos mensajes recientes
    st.subheader("Recent Posts")
    recent_posts = df_sentiment.sort_values('date', ascending=False).head(5)
    for _, row in recent_posts.iterrows():
        sentiment_color = "green" if row['score'] > 0.05 else "red" if row['score'] < -0.05 else "gray"
        st.markdown(f"""
        <div style='border-left: 3px solid {sentiment_color}; padding-left: 10px;'>
            <p style='font-size: 0.8em; color: gray;'>{row['date']}</p>
            <p>{row['text']}</p>
            <p style='font-size: 0.9em;'>Sentiment: {row['cat_sent']} | Score: {row['score']:.3f}</p>
        </div>
        <hr>
        """, unsafe_allow_html=True)
else:
    st.warning("No StockTwits data available for this ticker or time period.")

st.success("✅ Social Sentiment Analysis loaded. Next: AI Analysis.")
st.markdown("---")

def generate_ai_analysis(ticker, df_technical, df_news, df_sentiment):
    """
    Genera un análisis completo de la acción basado en datos técnicos,
    noticias y sentimiento social.
    """
    # 0) Diccionario de salida
    analysis = {
        "ticker": ticker,
        "overall_rating": 0,
        "recommendation": "MANTENER",
        "key_points": []
    }

    # —— 1. Análisis Técnico ——  
    technical_score  = 0
    technical_points = []

    if not df_technical.empty:
        # 1.1) Extraer últimos valores COMO SCALARS
        last_price       = df_technical['Close'].iloc[-1]
        last_sma20       = df_technical['SMA20'].iloc[-1]
        last_rsi         = df_technical['RSI'].iloc[-1]
        last_macd        = df_technical['MACD'].iloc[-1]
        last_signal      = df_technical['Signal Line'].iloc[-1]
        macd_signal_diff = last_macd - last_signal

        # 1.2) Cambio neto a 30 días (float)
        days_to_analyze = min(30, len(df_technical) - 1)
        if days_to_analyze > 0:
            price_30d_ago = df_technical['Close'].iloc[-days_to_analyze-1]
            if price_30d_ago != 0:
                price_change_30d = (last_price - price_30d_ago) / price_30d_ago * 100
            else:
                price_change_30d = 0.0
        else:
            price_change_30d = 0.0

        # 1.3) Precio vs SMA20 (float)
        if last_sma20 != 0:
            price_vs_sma20_pct = (last_price / last_sma20 - 1) * 100
        else:
            price_vs_sma20_pct = 0.0

        # 1.4) Scoring técnico
        # — Tendencia 30d
        if price_change_30d > 10:
            technical_score += 1
            technical_points.append(f"Subida {price_change_30d:.1f}% en {days_to_analyze}d (alcista fuerte)")
        elif price_change_30d < -10:
            technical_score -= 1
            technical_points.append(f"Caída {abs(price_change_30d):.1f}% en {days_to_analyze}d (bajista fuerte)")
        else:
            technical_points.append(f"Cambio moderado {price_change_30d:.1f}% en {days_to_analyze}d")

        # — Precio vs SMA20
        if price_vs_sma20_pct > 5:
            technical_score += 1
            technical_points.append(f"Precio {price_vs_sma20_pct:.1f}% por encima de SMA20 (alcista)")
        elif price_vs_sma20_pct < -5:
            technical_score -= 1
            technical_points.append(f"Precio {price_vs_sma20_pct:.1f}% por debajo de SMA20 (bajista)")

        # — RSI
        if last_rsi > 70:
            technical_score -= 2
            technical_points.append("RSI > 70 (sobrecompra)")
        elif last_rsi < 30:
            technical_score += 2
            technical_points.append("RSI < 30 (sobreventa)")
        else:
            technical_points.append(f"RSI en {last_rsi:.1f} (normal)")

        # — MACD
        if macd_signal_diff > 0:
            technical_score += 1
            technical_points.append("MACD > Signal (alcista)")
        else:
            technical_score -= 1
            technical_points.append("MACD < Signal (bajista)")

        # 1.5) Guardar análisis técnico
        analysis["technical_analysis"] = {
            "score":               technical_score,
            "last_price":          last_price,
            "rsi":                 last_rsi,
            "macd_signal_diff":    macd_signal_diff,
            "price_change_30d":    price_change_30d,
            "price_vs_sma20_pct":  price_vs_sma20_pct,
            "key_points":          technical_points
        }
    else:
        analysis["technical_analysis"] = {
            "score": 0,
            "key_points": ["No hay datos técnicos disponibles"]
        }

    # —— 2. Análisis de Noticias ——  
    if df_news is not None and not df_news.empty:
        news_count = len(df_news)
        news_points = [f"Analizadas {news_count} noticias recientes"]
        recent = df_news['headline'].iloc[:5].tolist()
        news_points += [f"Titular: {h}" for h in recent]
        analysis["news_analysis"] = {
            "news_count": news_count,
            "recent_headlines": recent,
            "key_points": news_points
        }
    else:
        analysis["news_analysis"] = {
            "news_count": 0,
            "recent_headlines": [],
            "key_points": ["No hay noticias disponibles"]
        }

    # —— 3. Análisis de Sentimiento ——  
    sentiment_score = 0
    sentiment_points = []
    if df_sentiment is not None and not df_sentiment.empty:
        avg_sent = df_sentiment['score'].mean()
        sentiment_points.append(f"Sentimiento medio: {avg_sent:.3f}")
        # categorías si existen
        if 'cat_sent' in df_sentiment:
            bulls = df_sentiment['cat_sent'].str.lower().str.contains('bull').sum()
            bears = df_sentiment['cat_sent'].str.lower().str.contains('bear').sum()
        else:
            bulls = (df_sentiment['score'] > 0.2).sum()
            bears = (df_sentiment['score'] < -0.2).sum()
        ratio = bulls / max(1, bears)
        sentiment_points.append(f"Ratio bull/bear: {ratio:.1f}")
        # asignar puntuación
        if avg_sent > 0.3:
            sentiment_score += 2
        elif avg_sent > 0.1:
            sentiment_score += 1
        elif avg_sent < -0.3:
            sentiment_score -= 2
        elif avg_sent < -0.1:
            sentiment_score -= 1

        analysis["sentiment_analysis"] = {
            "score": sentiment_score,
            "avg_sentiment": avg_sent,
            "bull_bear_ratio": ratio,
            "key_points": sentiment_points
        }
    else:
        analysis["sentiment_analysis"] = {
            "score": 0,
            "key_points": ["No hay datos de sentimiento social"]
        }

    # —— 4. Puntuación Global y Recomendación ——  
    tech_score = analysis["technical_analysis"]["score"]
    sent_score = analysis["sentiment_analysis"]["score"]
    weighted_score = tech_score * 0.7 + sent_score * 0.3
    analysis["overall_rating"] = weighted_score

    if weighted_score >= 2:
        analysis["recommendation"] = "COMPRAR"
    elif weighted_score <= -2:
        analysis["recommendation"] = "VENDER"
    else:
        analysis["recommendation"] = "MANTENER"

    # juntar puntos clave
    analysis["key_points"] = (
        analysis["technical_analysis"]["key_points"][:3]
        + analysis["news_analysis"]["key_points"][:2]
        + analysis["sentiment_analysis"]["key_points"][:2]
    )

    return analysis
