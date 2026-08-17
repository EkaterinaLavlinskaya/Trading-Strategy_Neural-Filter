import yfinance as yf
data = yf.download("BZ=F", period="20y", interval="1mo")
print(data.shape)

data.columns = data.columns.get_level_values(0)


print(data.columns.tolist())
print(data.head())

data["Next_Close"] = data["Close"].shift(-1)

# Целевая: 1 если следующий месяц выше текущего (лонг), иначе 0 (шорт)
data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)

data = data.dropna(subset=["Next_Close"])
print(data["Target"].value_counts())
print(data["Target"].value_counts())
print(data.shape)
print(data[["Close", "Target"]].tail(10))
print("Всего строк:", len(data))
print("Всего значений Target:", data["Target"].sum() + (1 - data["Target"]).sum())
print()
print(data["Target"].value_counts())
print()
print(data[["Close", "Target"]].head(10))
print(data[["Close", "Target"]].tail(10))
import pandas as pd


df = data.copy()

# 1. Доходность за прошлый месяц
df["Return_1m"] = df["Close"].pct_change()

# 2. Волатильность месяца
df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]

# 3. Положение цены в диапазоне месяца
df["Range_Position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])

# 4. Простые скользящие средние
df["SMA_3"] = df["Close"].rolling(3).mean()
df["SMA_6"] = df["Close"].rolling(6).mean()

# 5. Отклонение от тренда
df["Deviation_3"] = df["Close"] - df["SMA_3"]
df["Deviation_6"] = df["Close"] - df["SMA_6"]

# 6. Объём
df["Volume"] = df["Volume"]

# Убираем строки с NaN (первые 6 месяцев для SMA_6)
df = df.dropna()

print("Размер после создания фичей:", df.shape)
print()
print(df[["Close", "Return_1m", "Volatility", "Range_Position", "SMA_6", "Target"]].tail(10))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Фичи и целевая
feature_cols = ["Return_1m", "Volatility", "Range_Position", 
                "SMA_3", "SMA_6", "Deviation_3", "Deviation_6", "Volume"]

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

print("Accuracy по фолдам:", [round(a, 3) for a in acc_scores])
print("Средний Accuracy:", round(np.mean(acc_scores), 3))
print()
print("F1 по фолдам:", [round(f, 3) for f in f1_scores])
print("Средний F1:", round(np.mean(f1_scores), 3))
