"""
core/sgd.py
"""

from typing import Callable
import numpy as np

__all__ = [
    "SGDRegressor",
]

from core.sgd_variants import BaseOpt


class SGDRegressor:
    """
    Модель линейной/полиномиальной регрессии, обучаемая
    стохастическим градиентом или любым оптимизатором из core.sgd_variants
    """

    def __init__(
        self,
        lr_schedule: Callable[[int], float] | None = None,
        *,
        optimizer: BaseOpt | None = None,
        batch_size: int = 32,
        penalty: str | None = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1_000,
        fit_intercept: bool = True,
        clip_norm: float | None = None,
        init: str = "zeros",
        random_state: int | None = None,
    ):
        if optimizer is None and lr_schedule is None:
            raise ValueError("Нужно передать либо optimizer, либо lr_schedule")

        self.lr_schedule = lr_schedule
        self.optimizer = optimizer
        self.bs = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.clip_norm = clip_norm
        self.init = init
        self.rng = np.random.default_rng(random_state)

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X] if self.fit_intercept else X

    def _init_weights(self, n_features):
        if self.init == "zeros":
            return np.zeros(n_features)
        if self.init == "xavier":
            lim = np.sqrt(6 / n_features)
            return self.rng.uniform(-lim, lim, n_features)
        if self.init == "he":
            std = np.sqrt(2 / n_features)
            return self.rng.normal(0, std, n_features)
        raise ValueError("init ∈ {'zeros','xavier','he'}")

    def _reg_grad(self, w):
        if self.penalty is None or self.alpha == 0.0:
            return 0.0
        w_use = w.copy()
        if self.fit_intercept:
            w_use[0] = 0.0
        if self.penalty == "l1":
            return self.alpha * np.sign(w_use)
        if self.penalty == "l2":
            return self.alpha * w_use
        if self.penalty == "elastic":
            return self.alpha * (
                self.l1_ratio * np.sign(w_use) + (1 - self.l1_ratio) * w_use
            )
        raise ValueError(f"Unknown penalty {self.penalty}")

    def fit(self, X, y):
        X = self._add_intercept(np.asarray(X, float))
        y = np.asarray(y, float).ravel()

        self.w_ = self._init_weights(X.shape[1])
        if self.optimizer is not None:
            self.optimizer.reset()

        for k in range(self.max_iter):
            idx = self.rng.choice(len(X), self.bs, replace=False)
            Xb, yb = X[idx], y[idx]

            grad = 2 * Xb.T @ (Xb @ self.w_ - yb) / self.bs
            grad += self._reg_grad(self.w_)

            if self.clip_norm is not None:
                g_norm = np.linalg.norm(grad)
                if g_norm > self.clip_norm:
                    grad *= self.clip_norm / g_norm

            if self.optimizer is not None:
                self.w_ = self.optimizer.update(self.w_, grad, k)
            else:
                lr = self.lr_schedule(k)
                self.w_ -= lr * grad
        return self

    def predict(self, X):
        return self._add_intercept(np.asarray(X, float)) @ self.w_

    __call__ = predict
