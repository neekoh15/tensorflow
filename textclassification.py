import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# get data files
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Convertir etiquetas a números: ham -> 0, spam -> 1
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Tokenizar y paddear los mensajes
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data['message'])

train_sequences = tokenizer.texts_to_sequences(train_data['message'])
test_sequences = tokenizer.texts_to_sequences(test_data['message'])

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post')

# Construir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_padded, train_data['label'], epochs=30, validation_data=(test_padded, test_data['label']), verbose=2)

# Función de predicción
def predict_message(pred_text):
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
    prediction = model.predict(padded)
    if prediction[0] > 0.5:
        return [float(prediction[0]), "spam"]
    else:
        return [float(prediction[0]), "ham"]

pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)

# Código de prueba
def test_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        print(prediction)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()
