import requests
import yfinance as yf
import pandas as pd
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# 🔥 CONFIGURAZIONE TELEGRAM 🔥
TELEGRAM_TOKEN = "7668161656:AAG8tSGP6o2xS5e5vYR60SPoHXNMx-J2EDo"
CHAT_ID = "100353280"

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    return response.json()

# 📊 RECUPERA I DATI DI XAU/USD

def get_gold_data():
    try:
        gold = yf.Ticker("GC=F")  # Futures Oro
        df = gold.history(period="7d", interval="1h")  # Ultimi 7 giorni, 1h
        
        if df.empty:
            print("⚠️ Nessun dato disponibile da Yahoo Finance.")
            return None
        
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df["MACD"] = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9).macd()
        df["EMA50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
        df["EMA200"] = EMAIndicator(df["Close"], window=200).ema_indicator()
        df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        df["Volume"] = df["Volume"]
        return df
    
    except Exception as e:
        print(f"❌ Errore nel recupero dei dati: {e}")
        return None

# 📈 IDENTIFICAZIONE SUPPORTI E RESISTENZE

def get_support_resistance(df):
    recent_lows = df["Low"].rolling(window=10).min().iloc[-1]
    recent_highs = df["High"].rolling(window=10).max().iloc[-1]
    return recent_lows, recent_highs

# 🔍 CONTROLLO SEGNALE DI TRADING

def check_signals():
    print("🔍 Controllo segnali in corso...")
    df = get_gold_data()

    if df is None:
        send_telegram_message("⚠️ Errore nel recupero dei dati XAU/USD.")
        print("❌ Nessun dato ricevuto da Yahoo Finance!")
        return
    else:
        print("✅ Dati ricevuti correttamente!")
    
    last_row = df.iloc[-1]
    rsi = last_row["RSI"]
    macd = last_row["MACD"]
    ema50 = last_row["EMA50"]
    ema200 = last_row["EMA200"]
    atr = last_row["ATR"]
    volume = last_row["Volume"]
    price = last_row["Close"]
    
    support, resistance = get_support_resistance(df)
    
    signal = None
    
    if rsi < 30 and macd > 0 and price > support and price > ema50 and volume > 0:
        signal = f"🔵 **BUY XAU/USD**\n📊 RSI: {rsi:.2f}, MACD rialzista\n📉 Supporto: {support:.2f}\n📈 EMA50: {ema50:.2f}, EMA200: {ema200:.2f}\n🎯 Target: {resistance:.2f}, Stop Loss: {support - atr:.2f}"
    
    elif rsi > 70 and macd < 0 and price < resistance and price < ema50 and volume > 0:
        signal = f"🔴 **SELL XAU/USD**\n📊 RSI: {rsi:.2f}, MACD ribassista\n📈 Resistenza: {resistance:.2f}\n📉 EMA50: {ema50:.2f}, EMA200: {ema200:.2f}\n🎯 Target: {support:.2f}, Stop Loss: {resistance + atr:.2f}"
    
    if signal:
        print(f"📢 Segnale generato: {signal}")
        send_telegram_message(signal)
    else:
        print("⏳ Nessun segnale trovato in questo controllo.")

# 🚀 AVVIO DEL BOT
send_telegram_message("✅ **Bot di trading XAU/USD ULTRA ISTINTO attivo!**")
while True:
    check_signals()
    time.sleep(1800)  # Controlla il mercato ogni 30 minuti