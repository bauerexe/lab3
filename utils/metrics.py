# utils/metrics.py
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Any

# ------------------------------------------------------------------
#  Классические регрессионные метрики
# ------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-MSE."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R² (ближе к 1 — лучше)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


# короткий алиас «r2» для удобного импорта
r2 = r2_score


# ------------------------------------------------------------------
#  Вспомогательный агрегатор, если нужен подробный отчёт
# ------------------------------------------------------------------

def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Возвращает словарь со всеми основными метриками."""
    return {
        "mse":  mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }
