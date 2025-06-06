import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf

# Cargar dataset
df = pd.read_csv('sign_mnist_train.csv')
X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y = to_categorical(df['label'])

# Separar datos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Crear modelo CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')  # 26 letras (A-Z)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

# Guardar modelo
model.save('sign_model.h5')
