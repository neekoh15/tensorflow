import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Crear un modelo de regresión lineal
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X, y, epochs=100, verbose=0)

# Hacer predicciones
X_new = np.array([[0.8]])
y_pred = model.predict(X_new)

print("Predicción:", y_pred[0, 0])

# Visualizar los resultados
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal con TensorFlow')
plt.show()
