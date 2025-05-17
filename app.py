import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import finnhub

st.set_page_config("AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ––– Sidebar como FORMULARIO ––––––––––––––––––––––––––––––––––––––––––––––––
with st.sidebar.form("params"):
    st.header("Market Data Options")
    ticker     = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2024-04-15"))
    end_date   = st.date_input("End Date", pd.Timestamp.today())
    
    st.markdown("---")
    st.header("📰 News Options")
    days   = st.slider("Days of news history",    1, 7, 3)
    limit  = st.slider("Max articles to fetch", 10,100,30)
    
    # este botón disparará TODO el análisis
    analyze = st.form_submit_button("🔍 Analyze Stock")

# ––– Sólo ejecutar cuando el usuario haga clic ––––––––––––––––––––––––––––––––
if not analyze:
    st.info("👈 Ajusta tus parámetros y haz click en **Analyze Stock**")
    st.stop()

# ––– 1) Technical indicators ––––––––––––––––––––––––––––––––––––––––––––––––
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No data for '{ticker}'")
    st.stop()

df['SMA20'] = df['Close'].rolling(20).mean()
delta = df['Close'].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
rs    = gain.ewm(span=14).mean() / loss.ewm(span=14).mean()
df['RSI'] = 100 - (100 / (1 + rs))

st.subheader("1️⃣ Technical Indicators")
fig, ax = plt.subplots()
ax.plot(df.index, df['RSI'], label="RSI")
ax.legend()
st.pyplot(fig)

# ––– 2) News analysis via Finnhub ––––––––––––––––––––––––––––––––––––––––––––
api_key = st.secrets.get("FINNHUB_KEY")
if not api_key:
    st.warning("🔑 Please set your FINNHUB_KEY in Streamlit Secrets.")
    st.stop()

fh = finnhub.Client(api_key=api_key)
from_time = int((pd.Timestamp.today() - pd.Timedelta(days=days)).timestamp())
to_time   = int(pd.Timestamp.today().timestamp())

news = fh.general_news('general', min_id=None)
# filtrar manualmente por fecha y ticker si quieres…
df_news = pd.DataFrame(news)
if df_news.empty:
    st.warning("📰 No news found for that ticker/parameters.")
else:
    st.subheader("2️⃣ News Analysis")
    st.dataframe(df_news[['datetime','headline','source','url']].head(limit))

# ––– 3️⃣ (futuro) Social Media Sentiment… –––––––––––––––––––––––––––––––––––––
