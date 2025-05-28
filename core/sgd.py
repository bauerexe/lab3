"""core/sgd.py
Базовый стохастический градиентный спуск для линейной / полиномиальной
регрессии, совместимый с NumPy ≥ 2.0 (без np.asfarray).
"""

import numpy as np
from typing import Callable

__all__ = [
    "SGDRegressor",
]


class SGDRegressor:
    """Мини‑реализация стохастического ГД с L1/L2/ElasticNet.

    Параметры
    ---------
    lr_schedule : Callable[[int], float]
        Функция, возвращающая learning‑rate на k‑й итерации.
    batch_size : int
        Размер батча (1 ≤ bs ≤ |X|).
    penalty : {None, 'l1', 'l2', 'elastic'}
        Тип регуляризации.
    alpha : float
        Коэффициент перед штрафом.
    l1_ratio : float
        Доля L1 в ElasticNet.
    max_iter : int
        Число проходов по датасету.
    fit_intercept : bool
        Добавлять ли столбец «1» к X.
    random_state : int | None
        Для воспроизводимости.
    """

    def __init__(
        self,
        lr_schedule: Callable[[int], float],
        *,
        batch_size: int = 32,
        penalty: str | None = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1_000,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ):
        self.lr_schedule = lr_schedule
        self.bs = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X] if self.fit_intercept else X

    def _reg_grad(self, w: np.ndarray) -> np.ndarray:
        if self.penalty is None or self.alpha == 0.0:
            return 0.0
        if self.penalty == "l1":
            return self.alpha * np.sign(w)
        if self.penalty == "l2":
            return self.alpha * w
        if self.penalty == "elastic":
            return self.alpha * (
                self.l1_ratio * np.sign(w) + (1 - self.l1_ratio) * w
            )
        raise ValueError(f"Unknown penalty: {self.penalty}")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)  # совместимо с NumPy 2.0+
        y = np.asarray(y, dtype=float).ravel()

        X = self._add_intercept(X)
        self.w_ = np.zeros(X.shape[1])

        for k in range(self.max_iter):
            idx = self.rng.choice(len(X), self.bs, replace=False)
            Xb, yb = X[idx], y[idx]

            grad = 2 * Xb.T @ (Xb @ self.w_ - yb) / self.bs
            grad += self._reg_grad(self.w_)

            lr = self.lr_schedule(k)
            self.w_ -= lr * grad
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._add_intercept(np.asarray(X, dtype=float))
        return X @ self.w_

    # удобный короткий вызов
    __call__ = predict
