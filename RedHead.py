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
import praw


import praw

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)
NEWSAPI_KEY = "your_newsapi_key"


# --- CONFIG ---
st.set_page_config(page_title="RedHead AI", layout="wide")
st.title("📊 RedHead AI – Real-Time Reddit and Headline Sentiment with Financial Risk Analysis")

# --- USER INPUT ---
st.sidebar.header("User Controls")
ticker = st.sidebar.text_input("Stock Ticker (e.g., TSLA)", value="TSLA")
days_back = st.sidebar.slider("Days of history", 30, 365, 180)


# --- HELPERS ---
def get_price_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    data = yf.download(ticker, start=start, end=end)
    return data

import praw

def get_reddit_comments(ticker):
    comments = []
    try:
        subreddit = reddit.subreddit("stocks")
        for comment in subreddit.comments(limit=1000):
            if ticker.lower() in comment.body.lower():
                comments.append(comment.body)
    except Exception as e:
        st.warning(f"Reddit error: {e}")
    return comments

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
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit

    # Ensure needed columns are present
    required_cols = ["treatment", "lag_return", "volatility", "return_tomorrow"]
    if not all(col in data.columns for col in required_cols):
        return "❌ Required columns missing."

    # Drop rows with any missing values
    df = data[required_cols].dropna()

    # Define variables
    y = df["treatment"]
    X = df[["lag_return", "volatility"]]

    # Ensure numeric type and remove infinite values
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # Add constant term
    X = sm.add_constant(X)

    # Safety check: Enough variation?
    if len(np.unique(y)) < 2:
        return "❌ Not enough variation in treatment for IPW."

    # Fit logistic regression
    try:
        model = Logit(y, X).fit(disp=0)
    except Exception as e:
        return f"❌ Logit model failed: {e}"

    # Get predicted propensity scores
    ps = model.predict(X)

    # Calculate weights
    weights = np.where(y == 1, 1 / ps, 1 / (1 - ps))

    # ATE estimate
    treated = df.loc[X.index][y == 1]
    control = df.loc[X.index][y == 0]

    try:
        ate = (
            np.average(treated["return_tomorrow"], weights=weights[y == 1]) -
            np.average(control["return_tomorrow"], weights=weights[y == 0])
        )
    except Exception as e:
        return f"❌ Failed to calculate weighted ATE: {e}"

    return f"📊 IPW Estimate of ATE:\n  ATE = {ate:.5f}\n  Interpretation: Positive sentiment days impact next-day return by {ate * 100:.2f}%."

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
ipw_result = ipw_estimate(data)
st.text(ipw_result)

# --- Causal Forest ---
cf_ate = causal_forest_estimate(data)
st.write(f"🌳 Causal Forest ATE: {cf_ate:.4f} (Impact of sentiment using forest approximation)")
