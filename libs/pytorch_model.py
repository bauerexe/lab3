from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_uci
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def pytorch_model(X_train, X_test, y_train, y_test, optimize : str = "SGD"):

    # конвертируем данные в тензоры
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # модель линейной регрессии
    model = nn.Linear(X_train.shape[1], 1)

    # оптимизаторы
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.3),
        'Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Nesterov': optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True),
        'AdaGrad': optim.Adagrad(model.parameters(), lr=0.9999991),
        'RMSProp': optim.RMSprop(model.parameters(), lr=0.1),
        'Adam': optim.Adam(model.parameters(), lr=0.01),
    }

    criterion = nn.MSELoss()

    # пример обучения с SGD
    optimizer = optimizers[optimize]

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    # Включите модель в режим оценки (важно, если используются dropout или batchnorm)
    model.eval()

    # предсказание
    with torch.no_grad():  # чтобы не вычислять градиенты при предсказании
        y_pred_torch = model(X_test_torch)

    # конвертация тензора в numpy-массив
    y_pred = y_pred_torch.numpy()

    # Далее расчёт метрик
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}, MAE: {mae}, R²: {r2}')

    return model

