import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def tensorflow_manual_train(X_train, X_test, y_train, y_test, optimize, epochs=500, batch_size=64):
    # Конвертация в тензоры
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # Модель
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
    ])

    # Оптимизаторы
    optimizer_options = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=0.1),
        'Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Nesterov': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        'AdaGrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
        'RMSProp': tf.keras.optimizers.RMSprop(learning_rate=0.01),
        'Adam': tf.keras.optimizers.Adam(learning_rate=0.01),
    }
    optimizer = optimizer_options[optimize]

    loss_fn = tf.keras.losses.MeanSquaredError()

    # Один шаг обучения — компилируется в граф
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Цикл по эпохам
    n = X_train.shape[0]
    for epoch in range(epochs):
        # Перемешка вручную
        indices = tf.random.shuffle(tf.range(n))
        X_train = tf.gather(X_train, indices)
        y_train = tf.gather(y_train, indices)

        # Мини-батчи
        for i in range(0, n, batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            loss = train_step(x_batch, y_batch)

        # Можно печатать прогресс
        if epoch % 100 == 0:
            tf.print(f"Epoch {epoch}, Loss:", loss)

    # Предсказание и метрики
    y_pred = model(X_test, training=False).numpy().squeeze()
    y_test = y_test.squeeze()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{optimize}: MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}')

    def predictor(X_input):
        X_input_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)
        y_pred = model(X_input_tf, training=False).numpy()
        return y_pred.squeeze()

    return predictor
