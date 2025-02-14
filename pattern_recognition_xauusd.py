import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema

# 📊 Scarica i dati storici di XAU/USD

def get_gold_data(start, end):
    gold = yf.Ticker("GC=F")
    df = gold.history(start=start, end=end, interval="1h")
    if df.empty:
        print("⚠️ Nessun dato disponibile da Yahoo Finance.")
        return None
    return df

# 🔍 Funzione per rilevare massimi e minimi locali

def detect_patterns(df):
    df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=10)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=10)[0]]['Close']
    return df

# 📈 Grafico con pattern riconosciuti

def plot_patterns(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Prezzo XAU/USD', color='black')
    plt.scatter(df.index, df['min'], color='green', marker='^', label='Minimi Locali')
    plt.scatter(df.index, df['max'], color='red', marker='v', label='Massimi Locali')
    plt.legend()
    plt.title("Riconoscimento Pattern XAU/USD")
    plt.show()

# 🚀 Avvio del riconoscimento pattern
start_date = "2023-08-01"
end_date = "2024-02-01"
df = get_gold_data(start_date, end_date)
if df is not None:
    df = detect_patterns(df)
    plot_patterns(df)
