import streamlit as st
import yfinance as yf

st.title("📈 Simple Stock Viewer")

ticker = st.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")

if ticker:
    data = yf.download(ticker, period="1mo", interval="1d")
    st.line_chart(data['Close'])
