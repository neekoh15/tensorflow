import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(-1, 1, 100)
y_train = 2 * X_train + np.random.randn(*X_train.shape) * 0.33


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

model.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50)

# Predecir valores
y_pred = model.predict(X_train)

# Graficar los datos y la línea de regresión
plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred, color='red')
plt.show()
