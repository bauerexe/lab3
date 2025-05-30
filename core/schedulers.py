"""
core/schedulers.py
"""

from abc import ABC, abstractmethod
import math

class BaseScheduler(ABC):
    """
    Базовый класс для расписаний learning-rate.

    :param lr0: float
        Начальное значение learning rate.
    :param name: str
        Имя расписания (для отладки/представления).
    """

    def __init__(self, lr0: float, name: str = "BaseScheduler"):
        self.lr0 = lr0
        self.name = name

    @abstractmethod
    def __call__(self, k: int) -> float:
        """
        Возвращает значение learning rate для шага k.

        :param k: int
            Номер шага (итерации) градиентного спуска.
        :return: float
            Значение learning rate на этом шаге.
        """
        ...

    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние (если у расписания есть циклы).
        По умолчанию — ничего не делает.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(lr0={self.lr0})"


class ConstantScheduler(BaseScheduler):
    """
    Постоянное расписание: f(k) = lr0.
    :param lr0: float
        Постоянное значение learning rate.
    """

    def __init__(self, lr0: float):
        super().__init__(lr0, name="ConstantScheduler")

    def __call__(self, k: int) -> float:
        return self.lr0


class ExponentialDecayScheduler(BaseScheduler):
    """
    Экспоненциальное затухание:
        f(k) = lr0 * exp(-lam * k)
    :param lr0: float
        Начальное значение learning rate.
    :param lam: float, default=1e-3
        Скорость затухания (λ).
    """

    def __init__(self, lr0: float, lam: float = 1e-3):
        super().__init__(lr0, name="ExponentialDecay")
        self.lam = lam

    def __call__(self, k: int) -> float:
        return self.lr0 * math.exp(-self.lam * k)


class PolynomialDecay(BaseScheduler):
    """
    Полиномиальное затухание:
        f(k) = lr0 / (1 + beta * k)^alpha
    :param lr0: float
        Начальное значение learning rate.
    :param alpha: float, default=0.5
        Степень затухания (α).
    :param beta: float, default=1.0
        Коэффициент роста знаменателя.
    """

    def __init__(self, lr0: float, alpha: float = .5, beta: float = 1.0):
        super().__init__(lr0, name="PolynomialDecay")
        self.alpha = alpha
        self.beta = beta

    def __call__(self, k: int) -> float:
        return self.lr0 / ((1.0 + self.beta * k) ** self.alpha)


class StepDecayScheduler(BaseScheduler):
    """
    Поэтапное затухание:
        f(k) = lr0 * gamma^{floor(k / step)}
    :param lr0: float
        Начальное значение learning rate.
    :param step: int, default=1000
        Число шагов между уменьшениями.
    :param gamma: float, default=0.5
        Мультипликатор затухания (γ).
    """

    def __init__(self, lr0: float, step: int = 1000, gamma: float = .5):
        super().__init__(lr0, name="StepDecay")
        self.step = step
        self.gamma = gamma

    def __call__(self, k: int) -> float:
        factor = k // self.step
        return self.lr0 * (self.gamma ** factor)


__all__ = [
    "BaseScheduler",
    "ConstantScheduler",
    "ExponentialDecayScheduler",
    "PolynomialDecay",
    "StepDecayScheduler",
]
