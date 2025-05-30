import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_loader import load_uci

from libs.pytorch_model import pytorch_model
from libs.keras_model import *


def choose_model(target : str = "quality" , model: str = 'torch', optimize: str = 'SGD'):
    X, y, _ = load_uci(
        "wine-quality",
        "winequality-red.csv",
        cache_dir="../data/uci_cache",
        target=target,  # Тут указали столбец для угадывания оставшиеся 11 колонок формируют матрицу X_raw
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model == 'torch':
        res_model = pytorch_model(X_train, X_test, y_train, y_test, optimize, batch_size=64)
        return lambda X_input: res_model(X_input), scaler
    elif model == 'tensorflow':
        res_model = tensorflow_manual_train(X_train, X_test, y_train, y_test, optimize)
        return lambda X_input: res_model(X_input), scaler


