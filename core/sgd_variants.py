"""
Дополнительные оптимизаторы: Momentum, RMSProp, Adam.
Каждый класс реализует .update(w, grad) и хранит своё внутреннее состояние.
Их можно передавать в SGDRegressor вместо lr_schedule.
"""
import numpy as np

class _BaseOpt:
    def __init__(self, lr=1e-2): self.lr = lr
    def update(self, w, grad, k): raise NotImplementedError

class SGD(_BaseOpt):
    def update(self, w, grad, k): return w - self.lr * grad

class Momentum(_BaseOpt):
    def __init__(self, lr=1e-3, beta=.9): super().__init__(lr); self.b, self.v = beta, 0
    def update(self, w, grad, k):
        self.v = self.b * self.v + (1 - self.b) * grad
        return w - self.lr * self.v

class RMSProp(_BaseOpt):
    def __init__(self, lr=1e-3, beta=.9, eps=1e-8):
        super().__init__(lr); self.b, self.eps, self.sq = beta, eps, 0
    def update(self, w, grad, k):
        self.sq = self.b * self.sq + (1 - self.b) * grad**2
        return w - self.lr * grad / (np.sqrt(self.sq) + self.eps)

class Adam(_BaseOpt):
    def __init__(self, lr=1e-3, b1=.9, b2=.999, eps=1e-8):
        super().__init__(lr); self.b1, self.b2, self.eps = b1, b2, eps
        self.m = self.v = 0
    def update(self, w, grad, k):
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad**2)
        m_hat = self.m / (1 - self.b1**(k+1))
        v_hat = self.v / (1 - self.b2**(k+1))
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
