"""
core/schedulers.py
~~~~~~~~~~~~~~~~~~
Learning-rate schedulers compatible with GradientDescent / SGD classes.
Каждый объект вызывается как lr = scheduler(k), где k — номер шага.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math

# --------------------------------------------------------------------------- #
#                               Базовый класс                                 #
# --------------------------------------------------------------------------- #

class BaseScheduler(ABC):
    """Абстрактный класс: реализует __call__(k) и reset()."""

    def __init__(self, lr0: float, name: str = "BaseScheduler"):
        self.lr0 = lr0
        self.name = name

    @abstractmethod
    def __call__(self, k: int) -> float: ...

    def reset(self) -> None:
        """Позволяет начать отсчёт шагов заново (если нужно)."""
        pass

    def __repr__(self) -> str:
        return f"{self.name}(lr0={self.lr0})"


# --------------------------------------------------------------------------- #
#                         Конкретные расписания                               #
# --------------------------------------------------------------------------- #

class ConstantScheduler(BaseScheduler):
    """f(k) = lr0"""

    def __init__(self, lr0: float):
        super().__init__(lr0, name="ConstantScheduler")

    def __call__(self, k: int) -> float:
        return self.lr0


class ExponentialDecayScheduler(BaseScheduler):
    """f(k) = lr0 · exp(-λ · k)"""

    def __init__(self, lr0: float, lam: float = 1e-3):
        super().__init__(lr0, name="ExponentialDecay")
        self.lam = lam

    def __call__(self, k: int) -> float:
        return self.lr0 * math.exp(-self.lam * k)


class PolynomialDecay(BaseScheduler):
    """f(k) = lr0 / (β·k + 1)^α"""

    def __init__(self, lr0: float, alpha: float = .5, beta: float = 1.0):
        super().__init__(lr0, name="PolynomialDecay")
        self.alpha, self.beta = alpha, beta

    def __call__(self, k: int) -> float:
        return self.lr0 / ((self.beta * k + 1.0) ** self.alpha)


class StepDecayScheduler(BaseScheduler):
    """f(k) = lr0 · γ^{⌊k / step⌋}"""

    def __init__(self, lr0: float, step: int = 1000, gamma: float = .5):
        super().__init__(lr0, name="StepDecay")
        self.step, self.gamma = step, gamma

    def __call__(self, k: int) -> float:
        return self.lr0 * (self.gamma ** (k // self.step))


class CosineAnnealingScheduler(BaseScheduler):
    """
    f(k) = lr_min + 0.5·(lr0 − lr_min)·(1 + cos(π·t / L))

    * первый цикл длиной T
    * все последующие — T_restart (если задан)
    """

    def __init__(self, lr0: float, T: int = 10_000,
                 lr_min: float = 0.0, T_restart: int | None = None):
        super().__init__(lr0, name="CosineAnnealing")
        self.T0 = T
        self.lr_min = lr_min
        self.T_restart = T_restart
        self._cycle_len = T
        self._t = 0

    def __call__(self, k: int) -> float:
        if self._t >= self._cycle_len:
            self._t = 0
            if self.T_restart is not None:
                self._cycle_len = self.T_restart

        lr = self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (
            1 + math.cos(math.pi * self._t / self._cycle_len)
        )
        self._t += 1
        return lr

    def reset(self):
        self._cycle_len = self.T0
        self._t = 0


# --------------------------------------------------------------------------- #
#                        Что экспортировать из модуля                         #
# --------------------------------------------------------------------------- #

__all__ = [
    "BaseScheduler",
    "ConstantScheduler",
    "ExponentialDecayScheduler",
    "PolynomialDecay",
    "StepDecayScheduler",
    "CosineAnnealingScheduler",
]
