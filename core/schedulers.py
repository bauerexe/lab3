"""
core/schedulers.py
"""

from abc import ABC, abstractmethod
import math

class BaseScheduler(ABC):
    def __init__(self, lr0: float, name: str = "BaseScheduler"):
        self.lr0 = lr0
        self.name = name

    @abstractmethod
    def __call__(self, k: int) -> float: ...

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.name}(lr0={self.lr0})"


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



__all__ = [
    "BaseScheduler",
    "ConstantScheduler",
    "ExponentialDecayScheduler",
    "PolynomialDecay",
    "StepDecayScheduler",
]
