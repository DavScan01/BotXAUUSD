import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import xgboost as xgb
import tradingview_ta
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📊 Titolo della Dashboard
st.set_page_config(page_title="Dashboard XAU/USD & GBP/JPY", layout="wide")
st.title("📈 Dashboard XAU/USD & GBP/JPY - Ultra Zeno AI")

# 📌 Selezione Asset e Timeframe
symbols_map = {"XAU/USD": "GC=F", "GBP/JPY": "GBPJPY=X"}
symbol_name = st.sidebar.selectbox("Scegli il mercato", list(symbols_map.keys()))
symbol = symbols_map[symbol_name]
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])

# 📡 Scarica i dati di mercato
@st.cache_data(ttl=900)  # Cache per 15 minuti
def get_market_data(symbol, period="7d", interval="1h"):
    asset = yf.Ticker(symbol)
    df = asset.history(period=period, interval=interval, auto_adjust=True)
    
    if df.empty:
        st.error(f"⚠️ Nessun dato disponibile per {symbol}.")
        return None
    return df

# 📈 Ottenere prezzo da TradingView
def get_tradingview_price(symbol):
    exchange = "OANDA" if symbol == "GC=F" else "FX_IDC"
    try:
        analysis = tradingview_ta.TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener="forex",
            interval="1h"
        )
        return analysis.get_analysis().indicators.get("close", None)
    except Exception:
        return None

# 🔍 Rilevare pattern e calcolare Stop Loss / Take Profit
def detect_patterns(df):
    df['volatility'] = df['Close'].pct_change().rolling(window=10).std().fillna(0.005)
    
    df['stop_loss_buy'] = df['Close'] * (1 - df['volatility'])
    df['take_profit_buy'] = df['Close'] * (1 + df['volatility'] * 2)
    df['stop_loss_sell'] = df['Close'] * (1 + df['volatility'])
    df['take_profit_sell'] = df['Close'] * (1 - df['volatility'] * 2)

    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                   df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    return df

# 🧠 Machine Learning per previsione trading
def train_market_model(df):
    df = detect_patterns(df.dropna())
    
    X = df[['Close', 'volatility', 'stop_loss_buy', 'take_profit_buy', 'stop_loss_sell', 'take_profit_sell']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    return model, scaler

# 🏆 Carica i dati e addestra il modello
df = get_market_data(symbol, period="7d", interval=timeframe)

if df is not None:
    df = detect_patterns(df)
    model, scaler = train_market_model(df)

    # 📊 Previsione AI per il prossimo segnale
    latest_data = df.iloc[-1:]
    X_latest = scaler.transform(latest_data[['Close', 'volatility', 'stop_loss_buy', 'take_profit_buy', 'stop_loss_sell', 'take_profit_sell']])
    prediction = model.predict(X_latest)[0]
    
    tradingview_price = get_tradingview_price(symbol)
    ai_signal = "BUY" if prediction == 1 else "SELL"

    # 🟢 Segnali di trading
    st.subheader(f"📢 Segnale AI: {ai_signal}")
    st.write(f"**Prezzo attuale Yahoo Finance:** {df['Close'].iloc[-1]:.2f}")
    if tradingview_price:
        st.write(f"**Prezzo TradingView:** {tradingview_price:.2f}")
    
    st.write(f"🔵 **Stop Loss (BUY):** {df['stop_loss_buy'].iloc[-1]:.2f}")
    st.write(f"🟢 **Take Profit (BUY):** {df['take_profit_buy'].iloc[-1]:.2f}")
    st.write(f"🟠 **Stop Loss (SELL):** {df['stop_loss_sell'].iloc[-1]:.2f}")
    st.write(f"🔻 **Take Profit (SELL):** {df['take_profit_sell'].iloc[-1]:.2f}")

    # 📊 Grafico interattivo
    st.subheader("📊 Grafico Candlestick")
    fig, ax = mpf.plot(df, type='candle', style='charles', volume=False, returnfig=True)
    st.pyplot(fig)

    # 📉 Indicatori Tecnici
    st.subheader("📈 Indicatori Tecnici")
    st.line_chart(df[['RSI', 'MACD']])
    
else:
    st.error("⚠️ Errore nel caricamento dei dati di mercato.")

