"""
core/sgd_variants.py
"""

import numpy as np

__all__ = [
    "BaseOpt",
    "SGD",
    "Momentum",
]

class BaseOpt:
    """
    Базовый класс для оптимизаторов градиентного спуска.

    :param lr: float, default=1e-2
        Базовый learning rate (η).
    :param weight_decay: float, default=0.0
        Коэффициент L2-регуляризации (λ), добавляется к градиенту.
    """

    def __init__(self, lr: float = 1e-2, weight_decay: float = 0.0):
        self.lr = lr
        self.wd = weight_decay

    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние оптимизатора (если есть).
        По умолчанию ничего не делает.
        """
        ...

    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:
        """
        Обновляет веса по правилу оптимизатора.

        :param w: np.ndarray, форма (n_features,)
            Текущий вектор весов.
        :param grad: np.ndarray, форма (n_features,)
            Градиент функции потерь по w.
        :param k: int
            Номер текущей итерации (начиная с 0).
        :return: np.ndarray, форма (n_features,)
            Новый вектор весов после шага оптимизации.
        """
        raise NotImplementedError


class SGD(BaseOpt):
    """
    «Ванильный» стохастический градиентный спуск.

    Правило обновления:
        w ← w − η (grad + λ w)

    Наследует lr и weight_decay из BaseOpt.
    """

    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:
        return w - self.lr * (grad + self.wd * w)


class Momentum(BaseOpt):
    """
    Стохастический градиентный спуск с импульсом.

    Правило обновления:
        v_t = β v_{t−1} + (1 − β) grad
        w ← w − η (v_t + λ w)

    :param lr: float, default=1e-2
        Learning rate η.
    :param beta: float, default=0.9
        Коэффициент импульса β.
    :param weight_decay: float, default=0.0
        Коэффициент L2-регуляции λ.
    """

    def __init__(self, lr: float = 1e-2, beta: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.v = 0.0

    def reset(self) -> None:
        """
        Сбрасывает накопленный импульс в v = 0.
        """
        self.v = 0.0

    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:
        """
        Выполняет шаг Momentum:

        :param w: np.ndarray, форма (n_features,)
            Текущий вектор весов.
        :param grad: np.ndarray, форма (n_features,)
            Градиент функции потерь по w.
        :param k: int
            Номер итерации (не используется в формуле).
        :return: np.ndarray, форма (n_features,)
            Обновлённый вектор весов.
        """
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return w - self.lr * (self.v + self.wd * w)
