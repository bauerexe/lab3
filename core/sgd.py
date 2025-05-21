# core/sgd.py
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Dict, List, Any
from .schedulers import ConstantScheduler   # ваш ConstantDecay можно оформить так
from .regularization import l2_penalty, l1_penalty, elastic_penalty
from pkg.gradient_descent import GradientDescent  # <- ваш базовый класс

class SGD(GradientDescent):
    """
    Stochastic (mini-batch) Gradient Descent.

    Parameters
    ----------
    learning_rate : float
        Базовый шаг (может изменяться scheduler-ом).
    batch_size : int
        1 ⇒ чистый SGD, len(dataset) ⇒ полный GD.
    scheduler : Callable[[int], float]
        Функция lr(k). По-умолчанию Constant.
    reg : Dict[str, float] | None
        {'l1': 1e-4, 'l2': 0.01}   — коэффициенты регуляризации.
    shuffle : bool
        Перемешивать ли выборку в начале каждой эпохи.
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        scheduler: Callable[[int], float] | None = None,
        reg: Dict[str, float] | None = None,
        shuffle: bool = True,
        max_iter: int = 20_000,
        name: str = "SGD"
    ):
        super().__init__(learning_rate, name=name, max_iter=max_iter)
        self.batch_size = batch_size
        self.scheduler = scheduler or ConstantScheduler(learning_rate)
        self.reg = reg or {}
        self.shuffle = shuffle

    def _unpack_x0(self, x0):
        """
        Разбирает x0, который должен быть либо
         - tuple: (X, y) или (X, y, w0)
         - dict: {'X':…, 'y':…, 'w0':…}
        и возвращает тройку (X, y, w0_or_None).
        """
        if isinstance(x0, tuple):
            # допускаем (X, y)  или  (X, y, w0)
            X, y, *rest = x0
            w0 = rest[0] if rest else None
        elif isinstance(x0, dict):
            X = x0["X"]
            y = x0["y"]
            w0 = x0.get("w0", None)
        else:
            raise ValueError(
                "Для SGD x0 должен быть tuple (X, y[, w0]) или dict {'X':…, 'y':…,'w0':…}"
            )
        return X, y, w0
    # --------------------------------------------------------------------- #
    #                   интерфейс, совместимый с GradientDescent            #
    # --------------------------------------------------------------------- #
    def schedule(self, k: int) -> float:
        return self.scheduler(k)

    def optimize(  # ← сигнатура совпадает с базой
            self,
            func,  # ← loss-function
            grad_func,  # ← gradient от loss-function
            x0,
            eps: float = 1e-8,
            **kwargs,
    ):
        import time
        # ---------- распаковка данных --------------------------------------
        if isinstance(x0, tuple):
            X, y, w0 = (*x0, None)[:3]  # допускаем (X, y) или (X, y, w0)
        elif isinstance(x0, dict):
            X = x0["X"];
            y = x0["y"];
            w0 = x0.get("w0")
        else:
            raise ValueError("x0 должен быть (X, y, [w0]) или словарём {'X':…, 'y':…}")

        # batch_size по приоритету: kwargs → self.batch_size → 32
        batch_size = kwargs.get("batch_size", getattr(self, "batch_size", 32))
        callback = kwargs.get("callback", None)

        n_samples, n_features = X.shape
        w = np.zeros((n_features, 1), dtype=np.float32) if w0 is None else w0.astype(np.float32, copy=True)

        history = []
        t0 = time.time()
        k_total = 0

        for epoch in range(self.max_iter):
            # ----------- shuffle -------------------------------------------
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            # ----------- mini-batch loop -----------------------------------
            for start in range(0, n_samples, batch_size):
                xb, yb = X[start:start + batch_size], y[start:start + batch_size]

                grad = grad_func(xb, yb, w)  # dL/dw  (n_features × 1)

                # -------- регуляризация --------------------------------
                if "l2" in self.reg:
                    grad += self.reg["l2"] * l2_penalty.grad(w)
                if "l1" in self.reg:
                    grad += self.reg["l1"] * l1_penalty.grad(w)
                if "elastic" in self.reg:
                    a, b = self.reg["elastic"]
                    grad += elastic_penalty.grad(w, a, b)

                lr = self.schedule(k_total)
                w -= lr * grad
                k_total += 1

                if np.linalg.norm(grad) < eps:
                    break
            else:
                # внешний цикл продолжается (не было break во внутреннем)
                loss_epoch = func(X, y, w)  # вычисляем loss на всей выборке
                history.append(loss_epoch)
                if callback:
                    callback(epoch, w, loss_epoch)
                continue
            break  # остановились по eps

        return w, history, k_total, time.time() - t0


# ------------------------ Модификации SGD ----------------------------------- #
class SGDMomentum(SGD):
    """SGD + классический momentum."""
    def __init__(self, momentum: float = 0.9, **kwargs):
        super().__init__(name="SGD-Momentum", **kwargs)
        self.mu = momentum
        self.v: np.ndarray | None = None

    def optimize(self, *args, **kwargs):
        # Переиспользуем всё из базового SGD, переопределяя только update
        import time
        (loss_fn, grad_fn, X, y, w0,
         eps, callback) = args[:7] + tuple(kwargs.values())[:0]  # распаковка позиционных

        n_samples, n_features = X.shape
        w = np.zeros((n_features, 1), dtype=np.float32) if w0 is None else w0.copy()
        self.v = np.zeros_like(w)

        history = []
        t0 = time.time()
        k_total = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                xb, yb = X[start:start+self.batch_size], y[start:start+self.batch_size]
                grad = grad_fn(xb, yb)

                # momentum
                self.v = self.mu * self.v + self.schedule(k_total) * grad
                w -= self.v
                k_total += 1

                if np.linalg.norm(grad) < eps:
                    break
            else:
                history.append(loss_fn(X, y))
                if callback:
                    callback(epoch, w, history[-1])
                continue
            break

        return w, history, k_total, time.time() - t0

