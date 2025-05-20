# 📊 AI Stock Analyzer – README

## Overview
The **AI Stock Analyzer** is a Streamlit web app that empowers users to analyze any publicly listed stock using both **technical indicators** and **real-time news sentiment analysis**. It also offers AI-powered interpretations and final investment suggestions tailored to different investor profiles.

---

## 🧩 Features

### 📈 Technical Analysis
- **RSI (Relative Strength Index)** – Identifies overbought or oversold conditions
- **SMA20 (Simple Moving Average)** – Reveals market trend direction
- **MACD (Moving Average Convergence Divergence)** – Highlights momentum shifts and potential buy/sell signals

### 📰 News Sentiment Analysis
- Pulls top headlines using **NewsAPI**
- Applies **VADER** sentiment scoring to titles and summaries
- Displays distribution and time-based sentiment charts

### 🤖 AI Stock Insight
- Uses **OpenAI's GPT model** to summarize the technical and sentiment data
- Offers a human-like financial analysis in plain English

### 📍 Final Recommendation
- AI evaluates all indicators and gives a final decision: **BUY**, **HOLD**, or **DON'T BUY**
- Tailored to user-selected investor type (Day Trader, Swing Trader, Long-Term Investor)

---

## ⚙️ Configuration
### Required API Keys
- `NEWSAPI_KEY`: Get one at [https://newsapi.org](https://newsapi.org)
- `OPENAI_API_KEY`: Get one at [https://platform.openai.com](https://platform.openai.com)

You can set these either in `.streamlit/secrets.toml` or enter your OpenAI API key manually in the sidebar.

### Python Requirements (requirements.txt)
```
streamlit
pandas
numpy
matplotlib
altair
yfinance
requests
vaderSentiment
openai>=1.0.0
```

---

## 🚀 How to Run
```bash
git clone https://github.com/yourusername/ai-stock-analyzer.git
cd ai-stock-analyzer
streamlit run app.py
```

---

## 📌 Author & Notes
Built by students passionate about finance, AI, and data visualization. Ideal for academic projects, investor tools, or personal finance dashboards.
