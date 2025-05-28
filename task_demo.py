#!/usr/bin/env python3
"""
Task Demo — пример устойчивой к переполнению линейной регрессии,
обучаемой стохастическим градиентным спуском (SGD) и Adam.

Версия с полной аннотацией типов (Python ≥ 3.10).
"""
from __future__ import annotations

from typing import Callable, Protocol, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_uci
from utils.metrics import mse, r2
from core.sgd import SGDRegressor
from core.sgd_variants import Adam

# ---------------------------------------------------------------------------
#  Тип‑протокол для оптимизаторов из core.sgd_variants
# ---------------------------------------------------------------------------
class Optimizer(Protocol):
    """Интерфейс объекта‑оптимизатора (Adam, Momentum, …)."""

    lr: float  # текущий learning‑rate

    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:  # noqa: D401,E501
        """Возвращает новые веса, скорректированные по градиенту."""


# ---------------------------------------------------------------------------
#  Вспомогательные schedule‑функции
# ---------------------------------------------------------------------------
def constant_lr(lr: float) -> Callable[[int], float]:
    """Функция‑расписание с постоянным шагом *lr*."""

    return lambda _k: lr


# ---------------------------------------------------------------------------
#  Адаптер SGDRegressor → внешний Optimizer (Adam, RMSProp …)
# ---------------------------------------------------------------------------
class OptRegressor(SGDRegressor):
    """SGDRegressor, где обновление весов делает объект‑оптимизатор."""

    opt: Optimizer

    def __init__(self, optimizer: Optimizer, **kw: Any):
        super().__init__(lr_schedule=lambda _k: optimizer.lr, **kw)
        self.opt = optimizer

    # переопределяем цикл обучения → используем self.opt.update
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptRegressor":  # type: ignore[override]
        X_f = np.asarray(X, dtype=float)
        y_f = np.asarray(y, dtype=float).ravel()
        X_f = self._add_intercept(X_f)

        n_samples, n_features = X_f.shape
        self.w_ = np.zeros(n_features)
        bs = min(self.bs, n_samples)

        for k in range(self.max_iter):
            idx = self.rng.choice(n_samples, bs, replace=False)
            Xb, yb = X_f[idx], y_f[idx]

            grad = 2 * Xb.T @ (Xb @ self.w_ - yb) / bs
            grad += self._reg_grad(self.w_)

            self.w_ = self.opt.update(self.w_, grad, k)
        return self


# ---------------------------------------------------------------------------
#  Утилита для печати отчёта
# ---------------------------------------------------------------------------

def print_report(
    name: str,
    model: SGDRegressor,
    Xtr: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
) -> None:
    """Красиво печатает MSE и R² для train / test."""

    y_pred_tr = model(Xtr)
    y_pred_te = model(Xte)
    print(
        f"{name:>18}:  "
        f"trainMSE={mse(ytr, y_pred_tr):.4f}  "
        f"testMSE={mse(yte, y_pred_te):.4f}  "
        f"R²={r2(yte, y_pred_te):.4f}",
    )


# ---------------------------------------------------------------------------
#  Основная логика
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа скрипта."""

    # 1. Загрузка датасета --------------------------------------------------
    X_raw, y_raw, _ = load_uci(
        "wine-quality",
        "winequality-red.csv",
        cache_dir="data/uci_cache",
    target="quality", # Тут указали столбец для угадывания оставшиеся 11 колонок формируют матрицу X_raw
    )

    # 2. Train/Test split ----------------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
    )

    # 3. Стандартизируем признаки ------------------------------------------
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # 4. Базовый SGD --------------------------------------------------------
    sgd = SGDRegressor(
        lr_schedule=constant_lr(1e-1),  # учебный шаг после стандартизации
        batch_size=64,
        penalty="l2",
        alpha=1e-3,
        max_iter=20_000,
        random_state=42,
    ).fit(Xtr_s, ytr)
    print_report("SGD 1e-1", sgd, Xtr_s, Xte_s, ytr, yte)

    # 5. SGD + Adam ---------------------------------------------------------
    adam_model = OptRegressor(
        optimizer=Adam(lr=5e-3),
        batch_size=64,
        penalty="elastic",
        alpha=1e-3,
        l1_ratio=0.7,
        max_iter=4_000,
        random_state=42,
    ).fit(Xtr_s, ytr)
    print_report("SGD + Adam", adam_model, Xtr_s, Xte_s, ytr, yte)

    # 6. Индивидуальный прогноз -------------------------------------------
    sample_raw = np.asarray(
        [[7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
    )
    sample = scaler.transform(sample_raw)
    print("\nSample prediction (Adam):", round(adam_model(sample).item(), 3))

    print("\nSample prediction (sgd):", round(sgd(sample).item(), 3))

if __name__ == "__main__":
    main()
