"""
Бэктест торговой стратегии.
Симуляция сделок по шорт-сигналам с разными вариантами выхода.
"""

import pandas as pd
import numpy as np
from strategy import load_data, calculate_indicators, generate_short_signals, add_stop_loss


def run_backtest(df, exit_mode="base"):
    """
    Симуляция сделок.
    
    exit_mode:
    - "base": выход по MACD/RSI (1 свеча)
    - "trailing": трейлинг-стоп с самого начала
    - "smart_trailing": трейлинг включается после 0.5 пункта прибыли
    """
    signals = df[df["Signal_Short"]].copy()
    results = []

    for idx, row in signals.iterrows():
        pos = df.index.get_loc(idx)

        entry = row["Close"]
        initial_stop = row["Stop_Short"]

        # Параметры
        activation_profit = 0.5
        trailing_distance = 0.3

        current_stop = initial_stop
        trailing_active = False
        lowest_price = entry

        exit_price = None
        exit_time = None
        exit_reason = None

        for future_pos in range(pos + 1, len(df)):
            future_row = df.iloc[future_pos]

            # Обновляем минимум
            if future_row["Low"] < lowest_price:
                lowest_price = future_row["Low"]

            # Логика выхода в зависимости от режима
            if exit_mode == "trailing":
                # Трейлинг с самого начала
                new_stop = lowest_price + trailing_distance
                if new_stop < current_stop:
                    current_stop = new_stop

            elif exit_mode == "smart_trailing":
                # Умный трейлинг: включается после активации
                if not trailing_active:
                    profit_so_far = entry - lowest_price
                    if profit_so_far >= activation_profit:
                        trailing_active = True
                        current_stop = lowest_price + trailing_distance

                if trailing_active:
                    new_stop = lowest_price + trailing_distance
                    if new_stop < current_stop:
                        current_stop = new_stop

            # Стоп сработал?
            if future_row["High"] >= current_stop:
                exit_price = current_stop
                exit_time = df.index[future_pos]
                exit_reason = "trailing" if trailing_active else "initial_stop"
                break

            # Выход по MACD/RSI
            if exit_mode in ["base", "trailing", "smart_trailing"]:
                if future_row["MACD"] > future_row["Signal"] or future_row["RSI"] < 30:
                    exit_price = future_row["Close"]
                    exit_time = df.index[future_pos]
                    exit_reason = "MACD/RSI"
                    break

        if exit_price is None:
            exit_price = df.iloc[-1]["Close"]
            exit_time = df.index[-1]
            exit_reason = "end"

        pnl = entry - exit_price

        results.append({
            "Вход": idx,
            "Цена входа": entry,
            "Стоп": initial_stop,
            "Выход": exit_time,
            "Цена выхода": exit_price,
            "Причина": exit_reason,
            "PnL": pnl
        })

    return pd.DataFrame(results)


def print_statistics(results_df):
    """Вывод статистики по сделкам."""
    print(f"{'='*70}")
    print(f"СТАТИСТИКА")
    print(f"{'='*70}")
    print(f"Всего сделок: {len(results_df)}")
    print(f"Прибыльных: {(results_df['PnL'] > 0).sum()}")
    print(f"Убыточных: {(results_df['PnL'] < 0).sum()}")
    print(f"Процент прибыльных: {(results_df['PnL'] > 0).mean() * 100:.1f}%")
    print(f"\nСуммарный PnL: {results_df['PnL'].sum():.2f} пунктов")
    print(f"Средняя прибыль: {results_df['PnL'].mean():.3f}")
    print(f"Максимальная прибыль: {results_df['PnL'].max():.3f}")
    print(f"Максимальный убыток: {results_df['PnL'].min():.3f}")
    print()
    print(results_df)


if __name__ == "__main__":
    # Загрузка и подготовка данных
    df = load_data()
    df = calculate_indicators(df)
    df = generate_short_signals(df)
    df = add_stop_loss(df)
    df = df.dropna(subset=["MA10", "MACD", "Signal", "RSI", "Stop_Short"])

    print(f"Данных: {len(df)} свечей")
    print(f"Шорт-сигналов: {df['Signal_Short'].sum()}")
    print()

    # Базовая версия
    results = run_backtest(df, exit_mode="base")
    print_statistics(results)
