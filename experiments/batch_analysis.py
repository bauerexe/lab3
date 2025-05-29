"""
experiments/batch_analysis.py
"""
from __future__ import annotations

import time, tracemalloc, argparse
from itertools import combinations_with_replacement
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

from core.sgd import SGDRegressor
from core.schedulers import ConstantScheduler
from utils.metrics import mse, r2

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini‑batch size sweep for SGDRegressor")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--degree", type=int, default=1)
    return p.parse_args()

def polynomial_expand(X: np.ndarray, degree: int = 2) -> np.ndarray:
    if degree <= 1:
        return X.copy()
    n_samples, n_features = X.shape
    combos = []
    for d in range(1, degree + 1):
        combos.extend(combinations_with_replacement(range(n_features), d))
    X_poly = np.empty((n_samples, len(combos)), float)
    for j, c in enumerate(combos):
        X_poly[:, j] = np.prod(X[:, c], axis=1)
    return X_poly


def estimate_flops(n_feat: int, bs: int, epochs: int, n_samples: int) -> int:
    steps = epochs * int(np.ceil(n_samples / bs))
    return steps * 4 * bs * n_feat

def run_batch_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_sizes: List[int],
    lr_schedule,
    optimizer=None,
    epochs: int = 1000,
    poly_degree: int = 1,
    penalty: str | None = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    random_state: int | None = 42,
    plot: bool = False,
) -> Tuple[List[Dict[str, Any]], List[plt.Figure]]:

    if poly_degree > 1:
        X_train = polynomial_expand(X_train, poly_degree)
        X_test  = polynomial_expand(X_test,  poly_degree)

    n_samples, n_feat = X_train.shape
    results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        model = SGDRegressor(
            lr_schedule,
            optimizer=optimizer,
            batch_size=bs,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=epochs,
            random_state=random_state,
        )
        tracemalloc.start(); t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

        y_pred = model.predict(X_test)
        results.append({
            "batch": bs,
            "time_s": round(train_time,4),
            "peak_MB": round(peak/1024/1024,2),
            "mse": round(mse(y_test, y_pred),5),
            "r2":  round(r2(y_test, y_pred),5),
            "FLOPs": estimate_flops(n_feat, bs, epochs, n_samples),
        })

    figs: List[plt.Figure] = []
    if plot:
        df = pd.DataFrame(results).sort_values("batch")
        x = df["batch"]
        for col, title in [("mse", "Test MSE"), ("r2", "Test R²"), ("time_s", "Train time (s)")]:
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
    y = X @ rng.standard_normal(5) + rng.normal(0, 0.1, 1000)
    Xtr, Xte = X[:800], X[800:]
    ytr, yte = y[:800], y[800:]
    schedule = ConstantScheduler(1e-2)
    res, _ = run_batch_experiments(
        Xtr, ytr, Xte, yte,
        batch_sizes=[1,4,16,64,128, len(Xtr)],
        lr_schedule=schedule,
        epochs=args.epochs,
        poly_degree=args.degree,
        plot=False,
    )
    print(tabulate(res, headers="keys", tablefmt="github"))
