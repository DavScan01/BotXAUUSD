import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# 📊 Scarica i dati storici di XAU/USD e altri asset

def get_data(symbol, start, end):
    asset = yf.Ticker(symbol)
    df = asset.history(start=start, end=end, interval="1h")
    if df.empty:
        print(f"⚠️ Nessun dato per {symbol}")
        return None
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9).macd()
    df["EMA50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Close"], window=200).ema_indicator()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    return df

# 🔍 Backtesting su XAU/USD

from datetime import datetime

def backtest_xauusd(start="2023-08-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')  # Usa la data attuale se non specificata
    
    df = get_data("GC=F", start, end)
    if df is None:
        return
    
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trade_log = []
    
    for i in range(1, len(df)):
        rsi, macd, price, atr = df.iloc[i]["RSI"], df.iloc[i]["MACD"], df.iloc[i]["Close"], df.iloc[i]["ATR"]
        ema50, ema200 = df.iloc[i]["EMA50"], df.iloc[i]["EMA200"]
        
        # BUY Signal
        if rsi < 30 and macd > 0 and price > ema50:
            position = balance / price  # All-in trade
            balance = 0
            trade_log.append((df.index[i], "BUY", price))
            
        # SELL Signal
        elif rsi > 70 and macd < 0 and price < ema50 and position > 0:
            balance = position * price  # Chiudiamo trade
            position = 0
            trade_log.append((df.index[i], "SELL", price))
    
    final_balance = balance + (position * df.iloc[-1]["Close"]) if position > 0 else balance
    profit = final_balance - initial_balance
    
    print(f"📈 Backtesting Terminato. Profitto Finale: ${profit:.2f}")
    
    # 📊 Mostriamo i trade
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label="Prezzo XAU/USD", color="gray")
    buy_signals = [trade[2] for trade in trade_log if trade[1] == "BUY"]
    sell_signals = [trade[2] for trade in trade_log if trade[1] == "SELL"]
    buy_dates = [trade[0] for trade in trade_log if trade[1] == "BUY"]
    sell_dates = [trade[0] for trade in trade_log if trade[1] == "SELL"]
    plt.scatter(buy_dates, buy_signals, color='green', marker='^', label="BUY")
    plt.scatter(sell_dates, sell_signals, color='red', marker='v', label="SELL")
    plt.legend()
    plt.title("Backtesting XAU/USD con RSI e MACD")
    plt.show()

# 🔥 Avvia il backtest
backtest_xauusd()
