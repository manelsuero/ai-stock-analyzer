import streamlit as st
import pandas as pd
<<<<<<< Updated upstream
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
