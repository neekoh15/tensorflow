import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. Cargar el conjunto de datos IMDB
max_features = 10000  # Número de palabras más frecuentes a considerar
maxlen = 500  # Longitud máxima de las secuencias de entrada

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# 3. Preprocesar los datos
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# 4. Crear el modelo
model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

# 5. Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Entrenar el modelo
model.fit(train_data, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# 7. Evaluar el modelo
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Accuracy en el conjunto de prueba: {test_acc}')
