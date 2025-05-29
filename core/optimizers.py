"""
core/optimizers.py
~~~~~~~~~~~~~~~~~~

Единый набор оптимизаторов, совместимых с SGDRegressor:

* SGD (vanilla)                               * AdaGrad
* Momentum                                     * AdaDelta
* Nesterov Momentum                            * RMSProp
* Adam                                         * Adam + weight-decay
"""

from __future__ import annotations
import numpy as np


class BaseOpt:
    """Базовый интерфейс оптимизатора."""

    def __init__(self, lr: float = 1e-2, weight_decay: float = 0.0):
        self.lr = lr
        self.wd = weight_decay

    # инициализируем внутреннее состояние, если нужно
    def reset(self): ...

    # основной шаг оптимизации
    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------ #
#                         SGD-семейство                              #
# ------------------------------------------------------------------ #

class SGD(BaseOpt):
    def update(self, w, grad, k):
        return w - self.lr * (grad + self.wd * w)


class Momentum(BaseOpt):
    def __init__(self, lr=1e-3, beta=.9, **kw):
        super().__init__(lr, **kw)
        self.b, self.v = beta, 0

    def reset(self): self.v = 0

    def update(self, w, grad, k):
        self.v = self.b * self.v + (1 - self.b) * grad
        return w - self.lr * (self.v + self.wd * w)


class Nesterov(BaseOpt):
    """Momentum с предвзглядом («look-ahead»)."""

    def __init__(self, lr=1e-3, beta=.9, **kw):
        super().__init__(lr, **kw)
        self.b, self.v = beta, 0

    def reset(self): self.v = 0

    def update(self, w, grad, k):
        self.v = self.b * self.v + (1 - self.b) * grad
        look_ahead = w - self.lr * self.b * self.v
        return look_ahead - self.lr * (1 - self.b) * grad - self.lr * self.wd * w


# ------------------------------------------------------------------ #
#                       AdaGrad / AdaDelta / RMSProp                 #
# ------------------------------------------------------------------ #

class AdaGrad(BaseOpt):
    def __init__(self, lr=1e-2, eps=1e-8, **kw):
        super().__init__(lr, **kw)
        self.eps = eps
        self.G = 0  # аккумулируем квадраты

    def reset(self): self.G = 0

    def update(self, w, grad, k):
        self.G += grad ** 2
        return w - self.lr * grad / (np.sqrt(self.G) + self.eps) - self.lr * self.wd * w


class AdaDelta(BaseOpt):
    def __init__(self, rho=.95, eps=1e-6, **kw):
        super().__init__(lr=1.0, **kw)  # lr не нужен
        self.rho, self.eps = rho, eps
        self.Eg2 = 0
        self.Edx2 = 0

    def reset(self):
        self.Eg2 = self.Edx2 = 0

    def update(self, w, grad, k):
        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad ** 2
        RMS_g   = np.sqrt(self.Eg2 + self.eps)
        RMS_dx  = np.sqrt(self.Edx2 + self.eps)
        dx      = -(RMS_dx / RMS_g) * grad
        self.Edx2 = self.rho * self.Edx2 + (1 - self.rho) * dx ** 2
        return w + dx - self.wd * w


class RMSProp(BaseOpt):
    def __init__(self, lr=1e-3, beta=.9, eps=1e-8, **kw):
        super().__init__(lr, **kw)
        self.b, self.eps, self.sq = beta, eps, 0

    def reset(self): self.sq = 0

    def update(self, w, grad, k):
        self.sq = self.b * self.sq + (1 - self.b) * grad ** 2
        return w - self.lr * grad / (np.sqrt(self.sq) + self.eps) - self.lr * self.wd * w


# ------------------------------------------------------------------ #
#                                Adam                                #
# ------------------------------------------------------------------ #

class Adam(BaseOpt):
    def __init__(self, lr=1e-3, b1=.9, b2=.999, eps=1e-8, **kw):
        super().__init__(lr, **kw)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = self.v = 0

    def reset(self): self.m = self.v = 0

    def update(self, w, grad, k):
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat = self.m / (1 - self.b1 ** (k + 1))
        v_hat = self.v / (1 - self.b2 ** (k + 1))
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return w - step - self.lr * self.wd * w
