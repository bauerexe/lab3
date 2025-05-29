import numpy as np

class BaseOpt:
    def __init__(self, lr: float = 1e-2, weight_decay: float = 0.0):
        self.lr = lr
        self.wd = weight_decay

    def reset(self): ...

    def update(self, w: np.ndarray, grad: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


class SGD(BaseOpt):
    def update(self, w, grad, k):
        return w - self.lr * (grad + self.wd * w)

class Momentum(BaseOpt):
    def __init__(self, lr=1e-2, beta=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta, self.v = beta, 0.0

    def reset(self):
        self.v = 0.0

    def update(self, w, grad, k):
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return w - self.lr * (self.v + self.wd * w)

__all__ = ["BaseOpt", "SGD", "Momentum"]