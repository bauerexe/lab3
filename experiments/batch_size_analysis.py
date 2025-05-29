"""
experiments/batch_size_analysis.py
=================================

**Цель**
--------
Систематически измерить влияние размера мини‑батча на скорость, потребление
памяти и качество обучения линейной/полиномиальной регрессии, обучаемой
методом *Stochastic Gradient Descent*.

Разрабатывался как исправление недочётов пункта 1 лабораторной № 3.

Размещение
~~~~~~~~~~
- Поместите файл в каталог **`experiments/`** в корне репозитория:
  ``experiments/batch_size_analysis.py``.
- Запуск из терминала:

  ```bash
  python experiments/batch_size_analysis.py  # или
  python -m experiments.batch_size_analysis
  ```

- При необходимости его можно импортировать в ноутбуке:

  ```python
  from experiments.batch_size_analysis import run_batch_experiments
  ```

Функциональность
~~~~~~~~~~~~~~~~
- Перебирает список размеров батча и собирает:
  * время обучения (сек);
  * пиковое потребление RAM (МБ) — через ``tracemalloc``;
  * грубую оценку FLOPs;
  * метрики *MSE* и *R²* на тестовой выборке.
- Поддерживает *полиномиальное* расширение признаков любой степени без
  зависимости от scikit‑learn.
- Легко подключить произвольный ``lr_schedule`` и регуляризацию.

"""

from __future__ import annotations

import time
import tracemalloc
import argparse
from itertools import combinations_with_replacement
from typing import List, Dict, Any

import numpy as np
from tabulate import tabulate

# --- внутренние импорты проекта ------------------------------------------- #
from core.sgd import SGDRegressor
from core.schedulers import ConstantScheduler
from utils.metrics import mse, r2

# --------------------------------------------------------------------------- #
#                                                                        I/O #
# --------------------------------------------------------------------------- #


def _parse_cli() -> argparse.Namespace:
    """Простейший CLI: позволяет указать степень полинома и эпохи."""
    p = argparse.ArgumentParser(description="Mini‑batch size sweep for SGDRegressor")
    p.add_argument("--epochs", type=int, default=1000, help="Число эпох")
    p.add_argument("--degree", type=int, default=1, help="Степень полинома (1 = линейная регрессия)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#                                                           Feature helpers  #
# --------------------------------------------------------------------------- #


def polynomial_expand(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Простейшая реализация полиномиальных признаков (без зависимости sklearn).

    * degree = 1 → возврат копии X без изменений.
    * Включаются только *комбинации с повторением*, bias‑столбец не добавляется.
    """
    if degree <= 1:
        return X.copy()

    n_samples, n_features = X.shape
    combos = []
    for d in range(1, degree + 1):
        combos.extend(combinations_with_replacement(range(n_features), d))

    X_poly = np.empty((n_samples, len(combos)), dtype=float)
    for j, combo in enumerate(combos):
        X_poly[:, j] = np.prod(X[:, combo], axis=1)
    return X_poly


# --------------------------------------------------------------------------- #
#                                                         FLOP estimation    #
# --------------------------------------------------------------------------- #

def estimate_flops(
    n_features: int, batch_size: int, epochs: int, n_samples: int
) -> int:
    """Грубая оценка FLOP‑ов для SGD‑обновлений линейной регрессии.

    Предполагаем: в шаг входят две матричных операции (*forward* и *grad*),
    и каждая умножение‑сложение считается за 2 FLOP.
    """
    steps_per_epoch = int(np.ceil(n_samples / batch_size))
    total_steps = epochs * steps_per_epoch
    flops_per_step = 4 * batch_size * n_features
    return total_steps * flops_per_step


# --------------------------------------------------------------------------- #
#                                                      Основной эксперимент  #
# --------------------------------------------------------------------------- #

def run_batch_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_sizes: List[int],
    lr_schedule,
    epochs: int = 1000,
    poly_degree: int = 1,
    penalty: str | None = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    random_state: int | None = 42,
) -> List[Dict[str, Any]]:
    """Обучает SGDRegressor на разных размерах батча и возвращает отчёт."""

    # — при необходимости расширяем признаки —
    if poly_degree > 1:
        X_train = polynomial_expand(X_train, poly_degree)
        X_test = polynomial_expand(X_test, poly_degree)

    n_samples, n_features = X_train.shape
    results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        model = SGDRegressor(
            lr_schedule,
            batch_size=bs,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=epochs,
            random_state=random_state,
        )

        # — замеры времени и ОЗУ —
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # — инференс на тесте —
        y_pred = model.predict(X_test)

        results.append(
            {
                "batch": bs,
                "time_s": round(train_time, 4),
                "peak_MB": round(peak / 1024 / 1024, 2),
                "mse": round(mse(y_test, y_pred), 5),
                "r2": round(r2(y_test, y_pred), 5),
                "FLOPs": int(estimate_flops(n_features, bs, epochs, n_samples)),
            }
        )

    return results


# --------------------------------------------------------------------------- #
#                                                             Demo‑launch    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = _parse_cli()

    # ─── synthetic regression toy (можно заменить на реальный датасет) ─── #
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 5))
    w_true = rng.standard_normal(5)
    y = X @ w_true + rng.normal(scale=0.1, size=1000)

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    schedule = ConstantScheduler(lr0=1e-2)

    results = run_batch_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_sizes=[1, 8, 32, 128, len(X_train)],
        lr_schedule=schedule,
        epochs=args.epochs,
        poly_degree=args.degree,
    )

    print("\nBatch‑size sweep\n----------------")
    print(tabulate(results, headers="keys", tablefmt="github"))
