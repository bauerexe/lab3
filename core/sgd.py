"""
core/sgd.py
"""

from typing import Callable, Optional, Union
import numpy as np
from core.sgd_variants import BaseOpt

__all__ = [
    "SGDRegressor",
]

class SGDRegressor:
    """
    Модель линейной или полиномиальной регрессии,
    обучаемая стохастическим градиентным спуском (SGD)
    или любым оптимизатором из core.sgd_variants.

    Параметры:
    :param lr_schedule: Callable[[int], float] | None
        Функция расписания learning rate: принимает номер шага k и возвращает lrₖ.
    :param optimizer: BaseOpt | None
        Объект-оптимизатор (Momentum, Adam и т.п.). Если задан, используется вместо lr_schedule.
    :param batch_size: int
        Размер бача (количество случайно выбираемых примеров на шаге).
    :param penalty: str | None
        Тип регуляризации: "l1", "l2", "elastic" или None (без штрафа).
    :param alpha: float
        Сила регуляризации (λ).
    :param l1_ratio: float
        Доля L1 в Elastic-Net (только если penalty="elastic").
    :param max_iter: int
        Число шагов (итераций SGD).
    :param fit_intercept: bool
        Добавлять ли свободный член (intercept) в модель.
    :param clip_norm: float | None
        Граничное значение нормы градиента (gradient clipping), или None для отключения.
    :param init: str
        Метод инициализации весов: "zeros", "xavier" или "he".
    :param random_state: int | None
        Сид генератора случайных чисел для воспроизводимости.
    """

    def __init__(
        self,
        lr_schedule: Optional[Callable[[int], float]] = None,
        *,
        optimizer: Optional[BaseOpt] = None,
        batch_size: int = 32,
        penalty: Optional[str] = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1_000,
        fit_intercept: bool = True,
        clip_norm: Optional[float] = None,
        init: str = "zeros",
        random_state: Optional[int] = None,
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

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Добавляет столбец единиц в начале матрицы X, если fit_intercept=True.

        :param X: np.ndarray, форма (n_samples, n_features)
        :return: np.ndarray, форма (n_samples, n_features+1) — X с добавленным bias.
        """
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    def _init_weights(self, n_features: int) -> np.ndarray:
        """
        Инициализирует вектор весов длины n_features.

        :param n_features: int — число признаков (включая bias, если есть).
        :return: np.ndarray, форма (n_features,) — начальные веса.
        """
        if self.init == "zeros":
            return np.zeros(n_features)
        if self.init == "xavier":
            lim = np.sqrt(6.0 / n_features)
            return self.rng.uniform(-lim, lim, size=n_features)
        if self.init == "he":
            std = np.sqrt(2.0 / n_features)
            return self.rng.normal(0.0, std, size=n_features)
        raise ValueError("init ∈ {'zeros','xavier','he'}")

    def _reg_grad(self, w: np.ndarray) -> Union[np.ndarray, float]:
        """
        Вычисляет градиент регуляризационного штрафа для w.

        :param w: np.ndarray, форма (n_features,) — текущие веса.
        :return: np.ndarray или 0.0 — градиент регуляризации.
        """
        if self.penalty is None or self.alpha == 0.0:
            return 0.0
        w_corr = w.copy()
        if self.fit_intercept:
            w_corr[0] = 0.0
        if self.penalty == "l1":
            return self.alpha * np.sign(w_corr)
        if self.penalty == "l2":
            return self.alpha * w_corr
        if self.penalty == "elastic":
            return self.alpha * (
                self.l1_ratio * np.sign(w_corr) + (1 - self.l1_ratio) * w_corr
            )
        raise ValueError(f"Unknown penalty {self.penalty}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegressor":
        """
        Обучает модель методом SGD или внешним optimizer.

        :param X: np.ndarray, форма (n_samples, n_features) — матрица признаков.
        :param y: np.ndarray, форма (n_samples,) — вектор целей.
        :return: self — обученный объект SGDRegressor.
        """
        X_mat = self._add_intercept(np.asarray(X, dtype=float))
        y_vec = np.asarray(y, dtype=float).ravel()

        self.w_ = self._init_weights(X_mat.shape[1])
        if self.optimizer is not None:
            self.optimizer.reset()

        for k in range(self.max_iter):
            idx = self.rng.choice(len(X_mat), self.bs, replace=False)
            Xb, yb = X_mat[idx], y_vec[idx]

            grad = 2.0 * Xb.T @ (Xb @ self.w_ - yb) / self.bs
            grad += self._reg_grad(self.w_)

            if self.clip_norm is not None:
                norm = np.linalg.norm(grad)
                if norm > self.clip_norm:
                    grad *= (self.clip_norm / norm)

            if self.optimizer is not None:
                self.w_ = self.optimizer.update(self.w_, grad, k)
            else:
                lr = self.lr_schedule(k)
                self.w_ -= lr * grad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание y = X·w.

        :param X: np.ndarray, форма (n_samples, n_features)
        :return: np.ndarray, форма (n_samples,) — предсказанные значения.
        """
        X_mat = self._add_intercept(np.asarray(X, dtype=float))
        return X_mat @ self.w_

    __call__ = predict