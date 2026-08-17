import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

# Загрузка данных 
data = yf.download("BZ=F", period="20y", interval="1mo")
data.columns = data.columns.get_level_values(0)

#Целевая переменная
data["Next_Close"] = data["Close"].shift(-1)
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)
data = data.dropna(subset=["Next_Close"])

df = data.copy()

# Фичи
df["Return_1m"] = df["Close"].pct_change()
df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]
df["Range_Position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
df["SMA_3"] = df["Close"].rolling(3).mean()
df["SMA_6"] = df["Close"].rolling(6).mean()
df["Deviation_3"] = df["Close"] - df["SMA_3"]
df["Deviation_6"] = df["Close"] - df["SMA_6"]

df["Trend_3_6"] = (df["SMA_3"] - df["SMA_6"]) / df["SMA_6"]
df["Above_3m_High"] = (df["Close"] > df["Close"].rolling(3).max().shift(1)).astype(int)
df["Below_3m_Low"] = (df["Close"] < df["Close"].rolling(3).min().shift(1)).astype(int)
df["Bullish_Engulfing"] = ((df["Close"] > df["Open"]) & 
                            (df["Close"] > df["Open"].shift(1)) &
                            (df["Open"] < df["Close"].shift(1))).astype(int)
df["Bearish_Engulfing"] = ((df["Close"] < df["Open"]) & 
                            (df["Close"] < df["Open"].shift(1)) &
                            (df["Open"] > df["Close"].shift(1))).astype(int)

df = df.dropna()

#Модель
feature_cols = ["Return_1m", "Volatility", "Range_Position", 
                "SMA_3", "SMA_6", "Deviation_3", "Deviation_6", "Volume",
                "Trend_3_6", "Above_3m_High", "Below_3m_Low", 
                "Bullish_Engulfing", "Bearish_Engulfing"]

X = df[feature_cols]
y = df["Target"]

tscv = TimeSeriesSplit(n_splits=5)
acc_scores, f1_scores = [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

print("Размер данных:", df.shape)
print()
print("Accuracy по фолдам:", [round(a, 3) for a in acc_scores])
print("Средний Accuracy:", round(np.mean(acc_scores), 3))
print()
print("F1 по фолдам:", [round(f, 3) for f in f1_scores])
print("Средний F1:", round(np.mean(f1_scores), 3))
