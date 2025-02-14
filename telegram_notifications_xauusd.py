import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import tradingview_ta
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 🔥 CONFIGURAZIONE TELEGRAM 🔥
TELEGRAM_TOKEN = "7668161656:AAG8tSGP6o2xS5e5vYR60SPoHXNMx-J2EDo"
CHAT_ID = "100353280"

# 💽 Funzione per inviare messaggi su Telegram
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    requests.post(url, data=data)

# 📊 Ottenere il prezzo attuale da TradingView Web API
def get_tradingview_price(symbol):
    tradingview_url = f"https://www.tradingview.com/symbols/{symbol}/"
    try:
        response = requests.get(tradingview_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_element = soup.find('div', class_='tv-symbol-price-quote__value')
        if price_element:
            return float(price_element.text.replace(',', ''))
    except Exception as e:
        print(f"⚠️ Errore TradingView per {symbol}: {e}")
    return None

# 📊 Scarica i dati di mercato da Yahoo Finance
def get_market_data(symbol):
    yahoo_symbol = "GC=F" if symbol == "XAUUSD" else "GBPJPY=X"
    asset = yf.Ticker(yahoo_symbol)
    df = asset.history(period="1d", interval="1h", auto_adjust=True)
    if df.empty:
        return None
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# 🔍 Rileva pattern e calcola Stop Loss / Take Profit con ATR dinamico
def detect_patterns(df):
    df['ATR'] = df['Close'].rolling(14).std().fillna(0.005)
    df['stop_loss_buy'] = df['Close'] - (df['ATR'] * 1.5)
    df['take_profit_buy'] = df['Close'] + (df['ATR'] * 3)
    df['stop_loss_sell'] = df['Close'] + (df['ATR'] * 1.5)
    df['take_profit_sell'] = df['Close'] - (df['ATR'] * 3)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# 🤖 Machine Learning avanzato per previsione trading
def train_market_model(df):
    df = detect_patterns(df.dropna())
    if df.isnull().sum().sum() > 0 or df.empty:
        print("⚠️ Dati insufficienti per l'addestramento del modello.")
        return None, None
    X = df[['Close', 'ATR', 'stop_loss_buy', 'take_profit_buy', 'stop_loss_sell', 'take_profit_sell']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=300, eval_metric='logloss', max_depth=8, learning_rate=0.03)
    model.fit(X_train, y_train)
    return model, scaler

# 🛠️ Controlla i segnali e invia notifiche su Telegram
def check_and_notify_patterns(symbol, model, scaler):
    df = get_market_data(symbol)
    tradingview_price = get_tradingview_price(symbol)
    if tradingview_price:
        send_telegram_message(f"💰 *Prezzo attuale {symbol}: {tradingview_price}* (TradingView)")
    elif df is not None:
        price = df['Close'].iloc[-1]
        send_telegram_message(f"💰 *Prezzo attuale {symbol}: {price}* (Yahoo Finance)")
    else:
        return
    if df.isnull().sum().sum() > 0 or df.empty:
        print(f"⚠️ Dati insufficienti per generare segnali su {symbol}.")
        return
    df = detect_patterns(df)
    latest_data = df.iloc[-1:]
    try:
        X_latest = scaler.transform(latest_data[['Close', 'ATR', 'stop_loss_buy', 'take_profit_buy', 'stop_loss_sell', 'take_profit_sell']])
        prediction_proba = model.predict_proba(X_latest)[0][1]  # Probabilità BUY
    except ValueError:
        print(f"⚠️ Errore nella trasformazione dei dati per {symbol}.")
        return
    if prediction_proba >= 0.90:  # Segnale ultra-sicuro con probabilità >= 90%
        signal = 'BUY' if prediction_proba > 0.5 else 'SELL'
        message = f"""📢 *Segnale AI {symbol}: {signal}*
🛠️ Stop Loss (BUY): {df['stop_loss_buy'].iloc[-1]:.2f}
📈 Take Profit (BUY): {df['take_profit_buy'].iloc[-1]:.2f}
🛠️ Stop Loss (SELL): {df['stop_loss_sell'].iloc[-1]:.2f}
📉 Take Profit (SELL): {df['take_profit_sell'].iloc[-1]:.2f}
🎯 Probabilità: {prediction_proba*100:.2f}%"""
        send_telegram_message(message)

# ⚙️ Avvio del Bot
symbols = ["XAUUSD", "GBPJPY"]
df_market = get_market_data("XAUUSD")
if df_market is not None:
    model, scaler = train_market_model(df_market)
    if model is not None:
        send_telegram_message("✅ *Bot Ultra Zeno AI MAXIMUM PRECISION Attivo!* 🚀")
        while True:
            for symbol in symbols:
                check_and_notify_patterns(symbol, model, scaler)
            time.sleep(1800)  # Controlla ogni 30 minuti
    else:
        print("❌ Errore: Modello non addestrato a causa di dati insufficienti.")
else:
    print("❌ Errore: Nessun dato disponibile per avviare il bot.")

