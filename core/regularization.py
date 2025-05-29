"""
# core/regularization.py
"""

import numpy as np

class _L2Penalty:
    """‖w‖₂²"""

    @staticmethod
    def value(w: np.ndarray) -> float:
        return np.sum(w ** 2)

    @staticmethod
    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * w


class _L1Penalty:
    """‖w‖₁"""

    @staticmethod
    def value(w: np.ndarray) -> float:
        return np.sum(np.abs(w))

    @staticmethod
    def grad(w: np.ndarray) -> np.ndarray:
        return np.sign(w)


class _ElasticPenalty:
    """
    α · ‖w‖₂²  +  β · ‖w‖₁
    grad(w, α, β)
    """

    @staticmethod
    def value(w: np.ndarray, alpha: float, beta: float) -> float:
        return alpha * _L2Penalty.value(w) + beta * _L1Penalty.value(w)

    @staticmethod
    def grad(w: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha * _L2Penalty.grad(w) + beta * _L1Penalty.grad(w)

l2_penalty = _L2Penalty()
l1_penalty = _L1Penalty()
elastic_penalty = _ElasticPenalty()
