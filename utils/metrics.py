"""
utils/metrics.py
"""
from typing import Dict

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднеквадратичную ошибку (MSE).

    :param y_true: np.ndarray, shape (n_samples,)
        Истинные значения целевой переменной.
    :param y_pred: np.ndarray, shape (n_samples,)
        Предсказанные значения модели.
    :return: float
        Среднее значение (y_true - y_pred)^2.
    """
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет корень из среднеквадратичной ошибки (RMSE).

    :param y_true: np.ndarray, shape (n_samples,)
        Истинные значения целевой переменной.
    :param y_pred: np.ndarray, shape (n_samples,)
        Предсказанные значения модели.
    :return: float
        Корень из MSE.
    """
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE).

    :param y_true: np.ndarray, shape (n_samples,)
        Истинные значения целевой переменной.
    :param y_pred: np.ndarray, shape (n_samples,)
        Предсказанные значения модели.
    :return: float
        Среднее значение |y_true - y_pred|.
    """
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет коэффициент детерминации R².

    :param y_true: np.ndarray, shape (n_samples,)
        Истинные значения целевой переменной.
    :param y_pred: np.ndarray, shape (n_samples,)
        Предсказанные значения модели.
    :return: float
        R² = 1 - SS_res / SS_tot, где
        SS_res = sum((y_true - y_pred)^2),
        SS_tot = sum((y_true - mean(y_true))^2).
        Если SS_tot == 0, возвращает 0.0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

r2 = r2_score

def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Собирает полный набор метрик в словарь.

    :param y_true: np.ndarray, shape (n_samples,)
        Истинные значения целевой переменной.
    :param y_pred: np.ndarray, shape (n_samples,)
        Предсказанные значения модели.
    :return: Dict[str, float]
        Словарь с ключами:
        - 'mse': среднеквадратичная ошибка,
        - 'rmse': корень из MSE,
        - 'mae': средняя абсолютная ошибка,
        - 'r2': коэффициент детерминации.
    """
    return {
        "mse":  mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }