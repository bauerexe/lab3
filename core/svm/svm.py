"""
core/svm/svm.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

__all__ = [
    "train_linear_svm",
    "plot_svm_boundary",
]


def train_linear_svm(
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        random_state: int | None = None
) -> svm.SVC:
    """
    Обучает линейный SVM (SVC с kernel='linear') на данных (X, y).

    Параметры:
    :param X: np.ndarray, форма (n_samples, n_features) — матрица признаков.
    :param y: np.ndarray, форма (n_samples,) — вектор меток классов (0 или 1).
    :param C: float — параметр регуляризации (trade-off между шириной маржина и ошибками).
    :param random_state: int | None — сид для воспроизводимости обучения.
    :return: svm.SVC — обученная модель линейного SVM.
    """

    model = svm.SVC(kernel='linear', C=C, random_state=random_state)
    model.fit(X, y)
    return model


def plot_svm_boundary(
        model: svm.SVC,
        X: np.ndarray,
        y: np.ndarray,
        xrange: tuple[float, float] = None,
        yrange: tuple[float, float] = None,
        resolution: int = 200
) -> None:
    """
    Визуализирует разделяющую гиперплоскость и марджины линейного SVM.

    :param model: svm.SVC — обученная модель SVM с kernel='linear'.
    :param X: np.ndarray, форма (n_samples, 2) — двумерные данные для визуализации.
    :param y: np.ndarray, форма (n_samples,) — метки классов (0 или 1).
    :param xrange: tuple[float, float] | None — диапазон оси X для сетки (xmin, xmax).
    :param yrange: tuple[float, float] | None — диапазон оси Y для сетки (ymin, ymax).
    :param resolution: int — число точек по каждой оси для построения сетки.
    :return: None — отображает график с разделяющей линией и марджинами.
    """
    if xrange is None:
        xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    else:
        xmin, xmax = xrange
    if yrange is None:
        ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        ymin, ymax = yrange

    xx = np.linspace(xmin, xmax, resolution)
    yy = np.linspace(ymin, ymax, resolution)
    YY, XX = np.meshgrid(yy, xx)
    grid = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(grid).reshape(XX.shape)

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    plt.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear SVM: decision boundary and margins')
    plt.grid(True)
    plt.show()
