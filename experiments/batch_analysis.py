"""
experiments/batch_analysis.py

Утилита для исследования влияния размера мини-батча на скорость и качество обучения
SGDRegressor. Позволяет:
- Оценивать время обучения, пиковую память, MSE, R², FLOPs.
- По желанию строить графики зависимостей метрик от batch_size.
"""

import time
import tracemalloc
import argparse
from itertools import combinations_with_replacement
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

from core.sgd import SGDRegressor
from core.schedulers import ConstantScheduler
from utils.metrics import mse, r2

def _parse_cli() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    :return: Namespace c атрибутами:
        --epochs: int, число итераций обучения (default=1000),
        --degree: int, степень полиномиального расширения (default=1).
    """
    p = argparse.ArgumentParser(description="Mini-batch size sweep for SGDRegressor")
    p.add_argument("--epochs", type=int, default=1000,
                   help="Количество шагов SGD")
    p.add_argument("--degree", type=int, default=1,
                   help="Степень полиномиального расширения признаков")
    return p.parse_args()

def polynomial_expand(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Генерирует полиномиальные признаки заданной степени.

    :param X: np.ndarray, форма (n_samples, n_features) — исходная матрица признаков.
    :param degree: int — максимальная степень полинома (1 = без расширения).
    :return: np.ndarray, форма (n_samples, n_new_features) — расширенная матрица.
    """
    if degree <= 1:
        return X.copy()
    n_samples, n_features = X.shape
    combos: list[tuple[int, ...]] = []
    for d in range(1, degree + 1):
        combos.extend(combinations_with_replacement(range(n_features), d))
    X_poly = np.empty((n_samples, len(combos)), float)
    for j, combo in enumerate(combos):
        X_poly[:, j] = np.prod(X[:, combo], axis=1)
    return X_poly

def estimate_flops(n_feat: int, bs: int, epochs: int, n_samples: int) -> int:
    """
    Приблизительно оценивает количество FLoating Point Operations (FLOPs).

    :param n_feat: int — число признаков после расширения.
    :param bs: int — размер мини-бача.
    :param epochs: int — число итераций SGD.
    :param n_samples: int — общее число примеров.
    :return: int — оценка FLOPs: steps * 4 * bs * n_feat
    """
    steps = epochs * int(np.ceil(n_samples / bs))
    return steps * 4 * bs * n_feat

def run_batch_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_sizes: List[int],
    lr_schedule,
    optimizer: Optional[object] = None,
    epochs: int = 1000,
    poly_degree: int = 1,
    penalty: Optional[str] = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    random_state: Optional[int] = 42,
    plot: bool = False,
) -> Tuple[List[Dict[str, Any]], List[plt.Figure]]:
    """
    Выполняет серию запусков SGDRegressor с разными размерами батча и собирает метрики.

    :param X_train: np.ndarray, форма (n_train, n_features) — обучающая выборка.
    :param y_train: np.ndarray, форма (n_train,) — цели для обучающей выборки.
    :param X_test: np.ndarray, форма (n_test, n_features) — тестовая выборка.
    :param y_test: np.ndarray, форма (n_test,) — цели для тестовой выборки.
    :param batch_sizes: List[int] — список размеров батча для перебора.
    :param lr_schedule: Callable[[int], float] — функция изменения learning rate.
    :param optimizer: Optional[BaseOpt] — объект оптимизатора (Momentum, Adam и т.д.).
    :param epochs: int — число шагов обучения.
    :param poly_degree: int — степень полиномиального расширения признаков.
    :param penalty: Optional[str] — тип регуляризации ('l1','l2','elastic' или None).
    :param alpha: float — сила регуляризации λ.
    :param l1_ratio: float — доля L1 в Elastic-Net.
    :param random_state: Optional[int] — seed для воспроизводимости.
    :param plot: bool — если True, строятся графики метрик vs. batch_size.
    :return:
        results: List[Dict[str, Any]] — список словарей с метриками:
            'batch', 'time_s', 'peak_MB', 'mse', 'r2', 'FLOPs';
        figs: List[plt.Figure] — список фигур matplotlib, если plot=True.
    """
    # Полиномиальное расширение
    if poly_degree > 1:
        X_train = polynomial_expand(X_train, poly_degree)
        X_test = polynomial_expand(X_test, poly_degree)

    n_samples, n_feat = X_train.shape
    results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        model = SGDRegressor(
            lr_schedule=lr_schedule,
            optimizer=optimizer,
            batch_size=bs,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=epochs,
            random_state=random_state,
        )
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        y_pred = model.predict(X_test)
        results.append({
            "batch": bs,
            "time_s": round(train_time, 4),
            "peak_MB": round(peak / 1024 / 1024, 2),
            "mse": round(mse(y_test, y_pred), 5),
            "r2": round(r2(y_test, y_pred), 5),
            "FLOPs": estimate_flops(n_feat, bs, epochs, n_samples),
        })

    figs: List[plt.Figure] = []
    if plot:
        df = pd.DataFrame(results).sort_values("batch")
        x = df["batch"]
        for col, title in [
            ("mse", "Test MSE"),
            ("r2", "Test R²"),
            ("time_s", "Train time (s)")
        ]:
            fig, ax = plt.subplots()
            ax.plot(x, df[col], marker="o")
            ax.set_xscale("log")
            ax.set_xlabel("batch size (log)")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs batch size")
            ax.grid(True, ls=":")
            figs.append(fig)
        plt.show()

    return results, figs

if __name__ == "__main__":
    args = _parse_cli()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 5))
    y = X @ rng.standard_normal(5)
    Xtr, Xte = X[:800], X[800:]
    ytr, yte = y[:800], y[800:]
    schedule = ConstantScheduler(1e-2)
    res, _ = run_batch_experiments(
        Xtr, ytr, Xte, yte,
        batch_sizes=[1, 4, 16, 64, 128, len(Xtr)],
        lr_schedule=schedule,
        epochs=args.epochs,
        poly_degree=args.degree,
        plot=False,
    )
    print(tabulate(res, headers="keys", tablefmt="github"))
