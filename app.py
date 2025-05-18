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

# ── 5. Generar Análisis de IA ─────────────────────────────────────────────
# Función para generar análisis de IA
import pandas as pd

def generate_ai_analysis(ticker, df_technical, df_news, df_sentiment):
    """
    Genera un análisis basado en los datos técnicos, noticias y sentimiento social
    utilizando reglas predefinidas para simular un análisis de IA.
    """
    # Inicializar el resultado
    analysis = {
        "ticker": ticker,
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "technical_analysis": {},
        "news_analysis": {},
        "sentiment_analysis": {},
        "overall_rating": None,
        "recommendation": None,
        "key_points": []
    }
    
    # 1. Análisis técnico
    if not df_technical.empty:
        # — Últimos valores para indicadores clásicos
        last_price      = df_technical['Close'].iloc[-1]
        last_sma20      = df_technical['SMA20'].iloc[-1]
        last_rsi        = df_technical['RSI'].iloc[-1]
        last_macd       = df_technical['MACD'].iloc[-1]
        last_signal     = df_technical['Signal Line'].iloc[-1]
        macd_signal_diff = last_macd - last_signal
        
        # Definir cuántos días mirar atrás (hasta 30 o el máximo disponible)
        days_to_analyze = min(30, len(df_technical) - 1)
        
        # —— Cambio porcentual neto en 30 días (como float) ——
        if days_to_analyze > 0:
            price_30d_ago    = df_technical['Close'].iloc[-days_to_analyze-1]
            price_change_30d = (last_price - price_30d_ago) / price_30d_ago * 100
        else:
            price_change_30d = 0.0
        
        # Preparar scoring y puntos
        technical_score  = 0
        technical_points = []
        
        # RSI
        if last_rsi > 70:
            technical_score -= 2
            technical_points.append("RSI por encima de 70 sugiere sobrecompra")
        elif last_rsi < 30:
            technical_score += 2
            technical_points.append("RSI por debajo de 30 sugiere sobreventa")
        else:
            technical_points.append(f"RSI en {last_rsi:.1f}, dentro del rango normal")
        
        # MACD
        if macd_signal_diff > 0:
            technical_score += 1
            technical_points.append("MACD por encima de la línea de señal, posible tendencia alcista")
        else:
            technical_score -= 1
            technical_points.append("MACD por debajo de la línea de señal, posible tendencia bajista")
        
        # —— NUEVO: net change 30d ——  
        if price_change_30d > 10:
            technical_score += 1
            technical_points.append(
                f"Subida de {price_change_30d:.1f}% en los últimos {days_to_analyze} días (tendencia alcista fuerte)"
            )
        elif price_change_30d < -10:
            technical_score -= 1
            technical_points.append(
                f"Caída de {abs(price_change_30d):.1f}% en los últimos {days_to_analyze} días (tendencia bajista fuerte)"
            )
        else:
            technical_points.append(
                f"Cambio moderado: {price_change_30d:.1f}% en los últimos {days_to_analyze} días"
            )
        
        # Guardar el análisis técnico en el dict
        analysis["technical_analysis"] = {
            "score": technical_score,
            "last_price": last_price,
            "last_sma20": last_sma20,
            "rsi": last_rsi,
            "macd_signal_diff": macd_signal_diff,
            "price_change_30d": price_change_30d,
            "key_points": technical_points
        }
    
    # 2. Análisis de noticias
    if df_news is not None and not df_news.empty:
        # Número de noticias analizadas
        news_count = len(df_news)
        news_points = [f"Analizadas {news_count} noticias recientes"]
        
        # Simplemente noticias más recientes sin análisis de sentimiento
        recent_headlines = df_news['headline'].iloc[:5].tolist()
        news_points.extend([f"Titular reciente: {h}" for h in recent_headlines])
        
        analysis["news_analysis"] = {
            "news_count": news_count,
            "recent_headlines": recent_headlines,
            "key_points": news_points
        }
    
    # 3. Análisis de sentimiento social
    sentiment_score = 0
    sentiment_points = []
    
    if df_sentiment is not None and not df_sentiment.empty:
        # Calcular la puntuación media de sentimiento
        avg_sentiment = df_sentiment['score'].mean()
        sentiment_count = len(df_sentiment)
        
        # Contar sentimientos positivos y negativos
        if 'cat_sent' in df_sentiment.columns:
            bullish_count = df_sentiment['cat_sent'].str.lower().str.contains('bull').sum()
            bearish_count = df_sentiment['cat_sent'].str.lower().str.contains('bear').sum()
            bull_bear_ratio = bullish_count / max(1, bearish_count)
        else:
            # Si no tenemos categorías explícitas, inferir de las puntuaciones
            bullish_count = (df_sentiment['score'] > 0.2).sum()
            bearish_count = (df_sentiment['score'] < -0.2).sum()
            bull_bear_ratio = bullish_count / max(1, bearish_count)
        
        # Calcular puntuación de sentimiento
        if avg_sentiment > 0.3:
            sentiment_score += 2
            sentiment_points.append(f"Sentimiento social muy positivo ({avg_sentiment:.2f})")
        elif avg_sentiment > 0.1:
            sentiment_score += 1
            sentiment_points.append(f"Sentimiento social positivo ({avg_sentiment:.2f})")
        elif avg_sentiment < -0.3:
            sentiment_score -= 2
            sentiment_points.append(f"Sentimiento social muy negativo ({avg_sentiment:.2f})")
        elif avg_sentiment < -0.1:
            sentiment_score -= 1
            sentiment_points.append(f"Sentimiento social negativo ({avg_sentiment:.2f})")
        else:
            sentiment_points.append(f"Sentimiento social neutral ({avg_sentiment:.2f})")
        
        # Analizar ratio bull/bear
        if bull_bear_ratio > 2:
            sentiment_score += 1
            sentiment_points.append(f"Ratio alcista/bajista favorable: {bull_bear_ratio:.1f}")
        elif bull_bear_ratio < 0.5:
            sentiment_score -= 1
            sentiment_points.append(f"Ratio alcista/bajista desfavorable: {bull_bear_ratio:.1f}")
        
        # Guardar el análisis de sentimiento
        analysis["sentiment_analysis"] = {
            "score": sentiment_score,
            "avg_sentiment": avg_sentiment,
            "sentiment_count": sentiment_count,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "bull_bear_ratio": bull_bear_ratio,
            "key_points": sentiment_points
        }
    else:
        analysis["sentiment_analysis"] = {
            "score": 0,
            "key_points": ["No hay datos de sentimiento social disponibles"]
        }
    
    # 4. Generar puntuación global y recomendación
    # Combinar puntuaciones técnicas y de sentimiento (ponderadas)
    technical_weight = 0.7  # 70% peso para análisis técnico
    sentiment_weight = 0.3  # 30% peso para sentimiento
    
    technical_score = analysis.get("technical_analysis", {}).get("score", 0)
    weighted_score = (technical_score * technical_weight) + (sentiment_score * sentiment_weight)
    
    # Establecer la puntuación global
    analysis["overall_rating"] = weighted_score
    
    # Generar recomendación
    if weighted_score >= 2:
        analysis["recommendation"] = "COMPRAR"
        analysis["key_points"].append("Los indicadores técnicos y el sentimiento social apuntan a una tendencia alcista")
    elif weighted_score >= 0.5:
        analysis["recommendation"] = "MANTENER/COMPRAR"
        analysis["key_points"].append("Señales generalmente positivas, pero con algunas advertencias")
    elif weighted_score <= -2:
        analysis["recommendation"] = "VENDER"
        analysis["key_points"].append("Los indicadores técnicos y el sentimiento social apuntan a una tendencia bajista")
    elif weighted_score <= -0.5:
        analysis["recommendation"] = "MANTENER/VENDER"
        analysis["key_points"].append("Señales generalmente negativas, pero con algunas señales positivas")
    else:
        analysis["recommendation"] = "MANTENER"
        analysis["key_points"].append("Señales mixtas sin una tendencia clara")
    
    # Añadir puntos clave técnicos y de sentimiento
    analysis["key_points"].extend(analysis.get("technical_analysis", {}).get("key_points", [])[:3])
    analysis["key_points"].extend(analysis.get("sentiment_analysis", {}).get("key_points", [])[:2])
    
    return analysis

# Función para mostrar el análisis de IA
def display_ai_analysis(analysis):
    """
    Muestra el análisis de IA en una interfaz amigable de Streamlit
    """
    st.header("4️⃣ Análisis de IA")
    
    # Crear tres columnas para los indicadores principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Recomendación", 
            value=analysis["recommendation"],
            delta=f"{analysis['overall_rating']:.1f}" if analysis["overall_rating"] is not None else None,
            delta_color="normal"
        )
    
    with col2:
        # Mostrar puntuación técnica si está disponible
        if "technical_analysis" in analysis and "score" in analysis["technical_analysis"]:
            tech_score = analysis["technical_analysis"]["score"]
            st.metric(
                label="Puntuación Técnica", 
                value=f"{tech_score:.1f}",
                delta=None,
                delta_color="normal"
            )
    
    with col3:
        # Mostrar puntuación de sentimiento si está disponible
        if "sentiment_analysis" in analysis and "score" in analysis["sentiment_analysis"]:
            sent_score = analysis["sentiment_analysis"]["score"]
            st.metric(
                label="Puntuación de Sentimiento", 
                value=f"{sent_score:.1f}",
                delta=None,
                delta_color="normal"
            )
    
    # Crear una sección expandible para puntos clave
    with st.expander("📋 Puntos Clave", expanded=True):
        for i, point in enumerate(analysis["key_points"]):
            st.markdown(f"- {point}")
    
    # Crear secciones expandibles para cada tipo de análisis
    if "technical_analysis" in analysis and analysis["technical_analysis"]:
        with st.expander("📊 Detalles del Análisis Técnico", expanded=False):
            ta = analysis["technical_analysis"]
            
            # Mostrar indicadores numéricos claves
            st.markdown(f"**Último precio:** ${ta.get('last_price', 'N/A'):.2f}")
            st.markdown(f"**RSI (14):** {ta.get('rsi', 'N/A'):.1f}")
            
            # Precio vs SMA20
            price_vs_sma = ta.get('price_vs_sma20_pct')
            if price_vs_sma is not None:
                if price_vs_sma > 0:
                    st.markdown(f"**Precio vs SMA20:** +{price_vs_sma:.2f}% (por encima)")
                else:
                    st.markdown(f"**Precio vs SMA20:** {price_vs_sma:.2f}% (por debajo)")
            
            # MACD vs Signal
            macd_signal = ta.get('macd_signal')
            if macd_signal is not None:
                if macd_signal > 0:
                    st.markdown(f"**MACD vs Signal:** +{macd_signal:.4f} (cruce alcista)")
                else:
                    st.markdown(f"**MACD vs Signal:** {macd_signal:.4f} (cruce bajista)")
            
            # Variación 30 días
            price_change = ta.get('price_change_30d')
            if price_change is not None:
                st.markdown(f"**Variación 30 días:** {price_change:.2f}%")
    
    if "news_analysis" in analysis and analysis["news_analysis"]:
        with st.expander("📰 Resumen de Noticias", expanded=False):
            na = analysis["news_analysis"]
            
            st.markdown(f"**Noticias analizadas:** {na.get('news_count', 0)}")
            
            st.markdown("**Titulares recientes:**")
            for i, headline in enumerate(na.get('recent_headlines', [])[:5]):
                st.markdown(f"{i+1}. {headline}")
    
    if "sentiment_analysis" in analysis and analysis["sentiment_analysis"]:
        with st.expander("💬 Análisis de Sentimiento Social", expanded=False):
            sa = analysis["sentiment_analysis"]
            
            # Mostrar métricas clave
            st.markdown(f"**Sentimiento medio:** {sa.get('avg_sentiment', 0):.3f}")
            st.markdown(f"**Publicaciones analizadas:** {sa.get('sentiment_count', 0)}")
            
            # Mostrar ratio alcista/bajista
            bullish = sa.get('bullish_count', 0)
            bearish = sa.get('bearish_count', 0)
            ratio = sa.get('bull_bear_ratio', 0)
            
            st.markdown(f"**Publicaciones alcistas:** {bullish}")
            st.markdown(f"**Publicaciones bajistas:** {bearish}")
            st.markdown(f"**Ratio alcista/bajista:** {ratio:.2f}")
            
            # Interpretación del sentimiento
            if 'avg_sentiment' in sa:
                avg_sent = sa['avg_sentiment']
                if avg_sent > 0.3:
                    st.markdown("**Interpretación:** Sentimiento muy positivo")
                elif avg_sent > 0.1:
                    st.markdown("**Interpretación:** Sentimiento positivo")
                elif avg_sent < -0.3:
                    st.markdown("**Interpretación:** Sentimiento muy negativo")
                elif avg_sent < -0.1:
                    st.markdown("**Interpretación:** Sentimiento negativo")
                else:
                    st.markdown("**Interpretación:** Sentimiento neutral")

# Generar y mostrar el análisis de IA
with st.spinner("Generando análisis de IA..."):
    analysis = generate_ai_analysis(ticker, df, df_news, df_sentiment)
    display_ai_analysis(analysis)

st.markdown("---")
st.caption("Desarrollado por AI Stock Analyzer Team | Última actualización: Mayo 2025")
