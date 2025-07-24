# red_head_sentiment_full_app.py
# 📊 RedHead AI - Full Python-native Version (no OpenAI, no R)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.api import Logit, add_constant
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="RedHead AI", layout="wide")
st.title("📊 RedHead AI – Real-Time Reddit and Headline Sentiment with Financial Risk Analysis")

# --- USER INPUT ---
st.sidebar.header("User Controls")
ticker = st.sidebar.text_input("Stock Ticker (e.g., TSLA)", value="TSLA")
days_back = st.sidebar.slider("Days of history", 30, 365, 180)

REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_SECRET = "your_reddit_secret"
REDDIT_USER_AGENT = "your_user_agent"
NEWSAPI_KEY = "your_newsapi_key"

# --- HELPERS ---
def get_price_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    data = yf.download(ticker, start=start, end=end)
    return data

def get_reddit_comments(ticker):
    url = f"https://api.pushshift.io/reddit/comment/search?q={ticker}&subreddit=stocks&size=100"
    try:
        response = requests.get(url)
        results = response.json()
        return [c['body'] for c in results['data'] if 'body' in c]
    except:
        return []

def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(t)['compound'] for t in texts]
    df = pd.DataFrame({'text': texts, 'score': sentiments})
    return df

def run_garch(prices):
    returns = 100 * prices.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=5)
    return res, forecast

def ipw_estimate(data):
    data = data.dropna(subset=['treatment', 'lag_return', 'volatility', 'return_tomorrow'])
    X = add_constant(data[['lag_return', 'volatility']])
    y = data['treatment']
    model = Logit(y, X).fit(disp=0)
    p = model.predict(X)
    data['weight'] = np.where(data['treatment'] == 1, 1 / p, 1 / (1 - p))
    ate = (data[data['treatment'] == 1]['return_tomorrow'] * data[data['treatment'] == 1]['weight']).mean() - \
          (data[data['treatment'] == 0]['return_tomorrow'] * data[data['treatment'] == 0]['weight']).mean()
    return ate

def causal_forest_estimate(data):
    data = data.dropna(subset=['treatment', 'lag_return', 'volatility', 'return_tomorrow'])
    X = data[['lag_return', 'volatility']]
    y = data['return_tomorrow']
    w = data['treatment']
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    ate = np.mean(y[w == 1] - y_pred[w == 1]) - np.mean(y[w == 0] - y_pred[w == 0])
    return ate

# --- LOAD DATA ---
with st.spinner("Loading price data..."):
    price_data = get_price_data(ticker, days_back)
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.get_level_values(0)
    if 'Close' not in price_data.columns:
        st.error("'Close' column missing from price data. Try a different ticker.")
        st.stop()

# --- REDDIT SENTIMENT ---
st.subheader("🧠 Reddit Sentiment Analysis")
comments = get_reddit_comments(ticker)
if comments:
    sentiment_df = analyze_sentiment(comments)
    sentiment_score = sentiment_df['score'].mean()
    sentiment_summary = sentiment_df['score'].apply(lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')
    st.write(sentiment_df['score'].describe())
    st.bar_chart(sentiment_summary.value_counts())
else:
    st.warning("No Reddit comments found.")
    sentiment_score = np.nan

# --- PLOTTING PRICES ---
st.subheader("📈 Stock Price")
st.line_chart(price_data['Close'])

# --- GARCH MODEL ---
st.subheader("📉 GARCH Volatility Forecast")
try:
    res, forecast = run_garch(price_data['Close'])
    st.line_chart(res.conditional_volatility)
    st.write("5-day forecasted variance:", forecast.variance.iloc[-1].values)
except Exception as e:
    st.warning(f"GARCH model failed: {e}")

# --- CAUSAL ANALYSIS PREP ---
st.subheader("🧪 Causal Analysis")
returns = price_data['Close'].pct_change().dropna()
data = pd.DataFrame({
    'date': returns.index,
    'return_tomorrow': returns.shift(-1),
    'lag_return': returns.shift(1),
    'volatility': returns.rolling(5).std(),
    'sentiment_score': sentiment_score
})
data['treatment'] = (data['sentiment_score'] > 0).astype(int) if not np.isnan(sentiment_score) else np.nan

# --- IPW ---
ipw_ate = ipw_estimate(data)
st.write(f"📊 IPW ATE: {ipw_ate:.4f} (Impact of positive sentiment on next-day return)")

# --- Causal Forest ---
cf_ate = causal_forest_estimate(data)
st.write(f"🌳 Causal Forest ATE: {cf_ate:.4f} (Impact of sentiment using forest approximation)")
