# core/sgd_variants.py
from __future__ import annotations
import numpy as np
from .sgd import SGD
from .regularization import l2_penalty, l1_penalty, elastic_penalty

class SGDMomentum(SGD):
    """
    SGD with classical momentum:
      v ← μ·v + lr·∇L(w)
      w ← w − v
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        scheduler=None,
        reg: dict[str, float] | None = None,
        shuffle: bool = True,
        momentum: float = 0.9,
        max_iter: int = 20_000,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            scheduler=scheduler,
            reg=reg,
            shuffle=shuffle,
            max_iter=max_iter,
            name="SGD-Momentum",
        )
        self.mu = momentum
        self.v: np.ndarray | None = None

    def optimize(self, func, grad_func, x0, eps: float = 1e-8, **kwargs):
        # сохраняем базовую unpack-логику
        X, y, w0 = self._unpack_x0(x0)
        n_samples, n_features = X.shape
        w = np.zeros((n_features,1),dtype=np.float32) if w0 is None else w0.copy()
        self.v = np.zeros_like(w)

        history: list[float] = []
        import time; t0 = time.time(); k_total = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                g = grad_func(xb, yb, w)

                # регуляризация
                if "l2" in self.reg:      g += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:      g += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a,b = self.reg["elastic"]; g += elastic_penalty.grad(w,a,b)

                lr = self.schedule(k_total)
                # momentum update
                self.v = self.mu * self.v + lr * g
                w -= self.v

                k_total += 1
                if np.linalg.norm(g) < eps:
                    break
            else:
                history.append(func(X, y, w))
                continue
            break

        return w, history, k_total, time.time() - t0


class SGDNesterov(SGDMomentum):
    """
    SGD with Nesterov momentum:
      v_prev = v
      v ← μ·v − lr·∇L(w + μ·v)
      w ← w + v − μ·v_prev
    """
    def __init__(self, **kwargs):
        super().__init__(name="SGD-Nesterov", **kwargs)

    def optimize(self, func, grad_func, x0, eps: float = 1e-8, **kwargs):
        X, y, w0 = self._unpack_x0(x0)
        n_samples, n_features = X.shape
        w = np.zeros((n_features,1),dtype=np.float32) if w0 is None else w0.copy()
        self.v = np.zeros_like(w)

        history, k_total = [], 0
        import time; t0 = time.time()

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                # lookahead
                w_ahead = w - self.mu * self.v
                g = grad_func(xb, yb, w_ahead)

                if "l2" in self.reg:      g += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:      g += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a,b = self.reg["elastic"]; g += elastic_penalty.grad(w,a,b)

                lr = self.schedule(k_total)
                v_prev = self.v.copy()
                self.v = self.mu * self.v + lr * g
                w += -self.mu * v_prev + (1 + self.mu) * self.v

                k_total += 1
                if np.linalg.norm(g) < eps:
                    break
            else:
                history.append(func(X, y, w))
                continue
            break

        return w, history, k_total, time.time() - t0


class AdaGrad(SGD):
    """
    Adaptive Gradient (AdaGrad):
      G ← G + g⊙g
      w ← w − lr * g / (sqrt(G) + ε)
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        scheduler=None,
        reg=None,
        shuffle=True,
        epsilon: float = 1e-8,
        max_iter: int = 20_000,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            scheduler=scheduler,
            reg=reg,
            shuffle=shuffle,
            max_iter=max_iter,
            name="AdaGrad",
        )
        self.epsilon = epsilon

    def optimize(self, func, grad_func, x0, eps: float = 1e-8, **kwargs):
        X, y, w0 = self._unpack_x0(x0)
        n_samples, n_features = X.shape
        w = np.zeros((n_features,1),dtype=np.float32) if w0 is None else w0.copy()
        G = np.zeros_like(w)

        history, k_total = [], 0
        import time; t0 = time.time()

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                g = grad_func(xb, yb, w)

                if "l2" in self.reg:      g += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:      g += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a,b = self.reg["elastic"]; g += elastic_penalty.grad(w,a,b)

                G += g * g
                adj_grad = g / (np.sqrt(G) + self.epsilon)
                lr = self.schedule(k_total)
                w -= lr * adj_grad

                k_total += 1
                if np.linalg.norm(g) < eps:
                    break
            else:
                history.append(func(X, y, w))
                continue
            break

        return w, history, k_total, time.time() - t0


class RMSProp(SGD):
    """
    RMSProp:
      E[g²] ← ρ·E[g²] + (1−ρ)·g⊙g
      w ← w − lr * g / (sqrt(E[g²]) + ε)
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        scheduler=None,
        reg=None,
        shuffle=True,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        max_iter: int = 20_000,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            scheduler=scheduler,
            reg=reg,
            shuffle=shuffle,
            max_iter=max_iter,
            name="RMSProp",
        )
        self.rho = rho
        self.epsilon = epsilon

    def optimize(self, func, grad_func, x0, eps: float = 1e-8, **kwargs):
        X, y, w0 = self._unpack_x0(x0)
        n_samples, n_features = X.shape
        w = np.zeros((n_features,1),dtype=np.float32) if w0 is None else w0.copy()
        E = np.zeros_like(w)

        history, k_total = [], 0
        import time; t0 = time.time()

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                g = grad_func(xb, yb, w)

                if "l2" in self.reg:      g += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:      g += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a,b = self.reg["elastic"]; g += elastic_penalty.grad(w,a,b)

                E = self.rho * E + (1 - self.rho) * (g * g)
                adj_grad = g / (np.sqrt(E) + self.epsilon)
                lr = self.schedule(k_total)
                w -= lr * adj_grad

                k_total += 1
                if np.linalg.norm(g) < eps:
                    break
            else:
                history.append(func(X, y, w))
                continue
            break

        return w, history, k_total, time.time() - t0


class Adam(SGD):
    """
    Adam optimizer:
      m ← β1·m + (1−β1)·g
      v ← β2·v + (1−β2)·g⊙g
      m̂ = m / (1−β1^t),  v̂ = v / (1−β2^t)
      w ← w − lr * m̂ / (sqrt(v̂)+ε)
    """
    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        scheduler=None,
        reg=None,
        shuffle=True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 20_000,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            scheduler=scheduler,
            reg=reg,
            shuffle=shuffle,
            max_iter=max_iter,
            name="Adam",
        )
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon

    def optimize(self, func, grad_func, x0, eps: float = 1e-8, **kwargs):
        X, y, w0 = self._unpack_x0(x0)
        n_samples, n_features = X.shape
        w = np.zeros((n_features,1),dtype=np.float32) if w0 is None else w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)

        history, k_total = [], 0
        import time; t0 = time.time()

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                g = grad_func(xb, yb, w)

                if "l2" in self.reg:      g += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:      g += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a,b = self.reg["elastic"]; g += elastic_penalty.grad(w,a,b)

                m = self.beta1*m + (1-self.beta1)*g
                v = self.beta2*v + (1-self.beta2)*(g*g)
                m_hat = m / (1 - self.beta1**(k_total+1))
                v_hat = v / (1 - self.beta2**(k_total+1))

                lr = self.schedule(k_total)
                w -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                k_total += 1
                if np.linalg.norm(g) < eps:
                    break
            else:
                history.append(func(X, y, w))
                continue
            break

        return w, history, k_total, time.time() - t0
