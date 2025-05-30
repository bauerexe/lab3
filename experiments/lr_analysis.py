"""
experiments/lr_analysis.py
"""

from typing import Callable

import time
import tracemalloc
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.sgd import SGDRegressor
from core.schedulers import (
    ConstantScheduler,
    ExponentialDecayScheduler,
    PolynomialDecay,
    StepDecayScheduler,
)
from utils.metrics import mse, r2


def _default_schedulers() -> Dict[str, Callable[[int], float]]:
    """
    Возвращает словарь со стандартными расписаниями learning rate.

    :return: Dict[str, Callable[[int], float]] — отображение имени → объект расписания.
    """
    return {
        "const": ConstantScheduler(1e-2),
        "exp":   ExponentialDecayScheduler(1e-2, lam=5e-4),
        "poly":  PolynomialDecay(1e-2, alpha=0.6, beta=1.0),
        "step":  StepDecayScheduler(1e-2, step=400, gamma=0.5),
    }


def _default_penalties() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает словарь с конфигурациями видов регуляризации.

    :return: Dict[str, Dict[str, Any]] — отображение имени → параметры регуляризации.
    """
    return {
        "none":    {},
        "l1":      {"alpha": 1e-2},
        "l2":      {"alpha": 1e-2},
        "elastic": {"alpha": 1e-2, "l1_ratio": 0.5},
    }


def run_lr_reg_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    schedulers: Optional[Dict[str, Callable[[int], float]]] = None,
    penalties: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 32,
    epochs: int = 6000,
    random_state: int = 42,
    plot: bool = False,
) -> Tuple[pd.DataFrame, List[plt.Figure]]:
    """
    Перебирает все комбинации расписаний и регуляризаций, обучает и оценивает модели.

    :param X_train: np.ndarray, форма (n_train, n_features) — обучающая выборка.
    :param y_train: np.ndarray, форма (n_train,) — цели для обучающей выборки.
    :param X_test: np.ndarray, форма (n_test, n_features) — тестовая выборка.
    :param y_test: np.ndarray, форма (n_test,) — цели для тестовой выборки.
    :param schedulers: Optional[Dict[str, Callable[[int], float]]]
        Словарь расписаний learning rate (имя → объект расписания).
        Если None, используются _default_schedulers().
    :param penalties: Optional[Dict[str, Dict[str, Any]]]
        Словарь настроек регуляризации (имя → параметры).
        Если None, используются _default_penalties().
    :param batch_size: int — размер мини-бача.
    :param epochs: int — число итераций SGD.
    :param random_state: int — сид для генератора случайных чисел.
    :param plot: bool — если True, строятся bar-графики R² и времени.
    :return:
        Tuple[
            pd.DataFrame,                # Таблица результатов со столбцами:
                                        # 'scheduler', 'penalty', 'time_s',
                                        # 'peak_MB', 'mse', 'r2'
            List[plt.Figure]             # Список фигур, если plot=True
        ]
    """
    schedulers = schedulers or _default_schedulers()
    penalties  = penalties  or _default_penalties()

    rows: List[Dict[str, Any]] = []
    for sch_name, sch_obj in schedulers.items():
        for pen_name, pen_kwargs in penalties.items():
            model = SGDRegressor(
                lr_schedule=sch_obj,
                batch_size=batch_size,
                max_iter=epochs,
                penalty=None if pen_name == "none" else pen_name,
                **pen_kwargs,
                random_state=random_state,
            )
            tracemalloc.start()
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            t_sec = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            y_pred = model.predict(X_test)
            rows.append({
                "scheduler": sch_name,
                "penalty": pen_name,
                "time_s": round(t_sec, 3),
                "peak_MB": round(peak / 1024 / 1024, 2),
                "mse": round(mse(y_test, y_pred), 5),
                "r2": round(r2(y_test, y_pred), 5),
            })

    df = pd.DataFrame(rows).sort_values(["scheduler", "penalty"])
    figs: List[plt.Figure] = []

    if plot:
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        df.pivot(index="scheduler", columns="penalty", values="r2") \
          .plot(kind="bar", ax=ax1)
        ax1.set_title("R² vs scheduler × penalty")
        ax1.set_ylabel("R²")
        ax1.grid(axis="y", linestyle=":")
        figs.append(fig1)

        # График времени
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        df.pivot(index="scheduler", columns="penalty", values="time_s") \
          .plot(kind="bar", ax=ax2)
        ax2.set_title("Training time (s)")
        ax2.set_ylabel("seconds")
        ax2.grid(axis="y", linestyle=":")
        figs.append(fig2)

        plt.show()

    print(df)
    return df, figs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("LR-scheduler & regularization sweep")
    parser.add_argument("--epochs", type=int, default=6000,
                        help="Число итераций SGD")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 8))
    y = X @ rng.standard_normal(8) + rng.normal(0, 0.25, 1500)
    Xtr, Xte = X[:1200], X[1200:]
    ytr, yte = y[:1200], y[1200:]

    df_results, _ = run_lr_reg_experiments(
        Xtr, ytr, Xte, yte,
        epochs=args.epochs,
        plot=True
    )
    print("\n", df_results.to_string(index=False))