"""
Торговая стратегия на основе дипломной работы.
Инструмент: Brent Crude Oil (BZ=F)
Таймфрейм: M5
Индикаторы: MA10, MACD, RSI
"""

import pandas as pd
import numpy as np
import yfinance as yf

# Загрузка данных нефти Brent
def load_data(period="60d", interval="5m"):
    
    data = yf.download("BZ=F", period=period, interval=interval)
    data.columns = data.columns.get_level_values(0)
    return data

# Расчёт технических индикаторов
def calculate_indicators(df):
    
    # MA10
    df["MA10"] = df["Close"].rolling(10).mean()

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def generate_short_signals(df, confirm_bars=2):
    """
    Генерация шорт-сигналов.
    
    Условия:
    1. Цена ниже MA10
    2. MACD ниже сигнальной линии
    3. RSI вышел из зоны перекупленности (пересек 70 вниз)
    4. Подтверждение: сигнал держится confirm_bars свечей
    """
    df["Price_Below_MA10"] = df["Close"] < df["MA10"]
    df["MACD_Bearish"] = df["MACD"] < df["Signal"]
    df["RSI_Exit_Overbought"] = (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)

    df["Signal_Short_instant"] = (
        df["Price_Below_MA10"] &
        df["MACD_Bearish"] &
        df["RSI_Exit_Overbought"]
    )

    df["Signal_Short"] = False
    for i in range(confirm_bars, len(df)):
        if (df["Signal_Short_instant"].iloc[i-confirm_bars] and 
            df["Price_Below_MA10"].iloc[i] and 
            df["MACD_Bearish"].iloc[i]):
            df.loc[df.index[i], "Signal_Short"] = True

    return df


def add_stop_loss(df, lookback=5):
    """Стоп-лосс: максимум за lookback свечей до сигнала."""
    df["Stop_Short"] = df["High"].rolling(lookback).max().shift(1)
    return df
