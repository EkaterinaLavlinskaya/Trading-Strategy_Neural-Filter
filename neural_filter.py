"""
Нейросеть-фильтр для оценки качества сигналов.
Попытка предсказать, будет ли сделка прибыльной.

Внимание: на 14 сделках модель не обучается (точность 42.9%).
Нужно 100+ сделок для обучения.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from strategy import load_data, calculate_indicators, generate_short_signals, add_stop_loss
from backtest import run_backtest


def build_trade_dataset(df, results_df):
    """Сборка датасета: фичи на момент входа + целевая (прибыль/убыток)."""
    signals = df[df["Signal_Short"]].copy()

    trade_features = []

    for idx, row in signals.iterrows():
        pos = df.index.get_loc(idx)

        features = {
            "Close": row["Close"],
            "MA10": row["MA10"],
            "MACD": row["MACD"],
            "Signal": row["Signal"],
            "RSI": row["RSI"],
            "Stop": row["Stop_Short"],
            "Distance_to_Stop": abs(row["Close"] - row["Stop_Short"]),
            "Volatility_5": df["Close"].iloc[max(0, pos-5):pos].std(),
            "Return_5": (row["Close"] - df["Close"].iloc[max(0, pos-5)]) / df["Close"].iloc[max(0, pos-5)],
            "Hour": idx.hour,
            "DayOfWeek": idx.dayofweek,
            "Price_vs_MA10": row["Close"] / row["MA10"] - 1,
        }

        trade_features.append(features)

    trade_df = pd.DataFrame(trade_features)
    trade_df["Target"] = (results_df["PnL"] > 0).astype(int)

    return trade_df


if __name__ == "__main__":
    # Подготовка данных
    df = load_data()
    df = calculate_indicators(df)
    df = generate_short_signals(df)
    df = add_stop_loss(df)
    df = df.dropna(subset=["MA10", "MACD", "Signal", "RSI", "Stop_Short"])

    # Бэктест
    results = run_backtest(df, exit_mode="base")

    # Датасет
    trade_df = build_trade_dataset(df, results)

    # Модель
    feature_cols = ["Close", "MA10", "MACD", "Signal", "RSI", "Stop",
                    "Distance_to_Stop", "Volatility_5", "Return_5",
                    "Hour", "DayOfWeek", "Price_vs_MA10"]

    X = trade_df[feature_cols]
    y = trade_df["Target"]

    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    loo = LeaveOneOut()

    predictions = []
    true_values = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        predictions.append(pred)
        true_values.append(y_test.iloc[0])

    accuracy = sum(np.array(predictions) == np.array(true_values)) / len(true_values)
    print(f"Точность модели: {accuracy * 100:.1f}%")
    print(f"(Случайное угадывание: 50%)")
