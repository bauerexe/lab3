from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def pytorch_model(X_train, X_test, y_train, y_test, optimize: str = "SGD", lr : float = 0.1, epochs: int = 4000):
    # конвертируем данные в тензоры
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # модель линейной регрессии
    model = nn.Linear(X_train.shape[1], 1)

    # оптимизаторы
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=lr),
        'Momentum': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'Nesterov': optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True),
        'AdaGrad': optim.Adagrad(model.parameters(), lr=lr),
        'RMSProp': optim.RMSprop(model.parameters(), lr=lr),
        'Adam': optim.Adam(model.parameters(), lr=lr),
    }

    criterion = nn.MSELoss()
    optimizer = optimizers[optimize]

    n_samples = X_train_torch.shape[0]

    # обучение по mini-batch
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    # включаем режим оценки
    model.eval()

    # предсказание
    with torch.no_grad():
        y_pred_torch = model(X_test_torch)

    y_pred = y_pred_torch.numpy()

    # метрики
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}')

    def predictor(X_input):
        X_input_torch = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            y_out = model(X_input_torch)
        return y_out.numpy().squeeze()

    return predictor
