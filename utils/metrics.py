# utils/metrics.py
from __future__ import annotations
import time
import tracemalloc
import numpy as np
from typing import Callable, Dict, Any, Tuple

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R^2
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def evaluate_optimizer(
    optimizer: Any,
    loss_fn: Callable[..., float],
    grad_fn: Callable[..., np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eps: float = 1e-8,
    **opt_kwargs
) -> Dict[str, Any]:
    """
    Запускает optimizer.optimize, собирает и возвращает:
     - финальные веса
     - train/test MSE, RMSE, MAE, R2
     - число итераций и время работы
     - пиковое потребление памяти

    Параметры
    ----------
    optimizer
        Объект с методом .optimize(loss_fn, grad_fn, x0=(X_train,y_train), eps=…, **opt_kwargs)
    loss_fn, grad_fn
        Функции для оптимизатора
    X_train, y_train, X_test, y_test
        Данные
    eps
        Критерий остановки
    opt_kwargs
        Любые другие параметры для optimize, например batch_size, scheduler, reg

    Returns
    -------
    metrics : dict
    """
    # Начнём профилирование памяти
    tracemalloc.start()

    t0 = time.time()
    w_opt, history, n_steps, t_opt = optimizer.optimize(
        loss_fn, grad_fn, x0=(X_train, y_train), eps=eps, **opt_kwargs
    )
    t_total = time.time() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Предсказания
    y_train_pred = X_train @ w_opt
    y_test_pred  = X_test  @ w_opt

    return {
        "w_opt": w_opt,
        "train_mse": mse(y_train, y_train_pred),
        "test_mse": mse(y_test, y_test_pred),
        "train_rmse": rmse(y_train, y_train_pred),
        "test_rmse": rmse(y_test, y_test_pred),
        "train_mae": mae(y_train, y_train_pred),
        "test_mae": mae(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "n_steps": n_steps,
        "time_opt": t_opt,
        "time_total": t_total,
        "mem_peak_bytes": peak,
        "history": history,
    }
