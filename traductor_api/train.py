import cv2, mediapipe as mp
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Input((30, 543*3)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu')),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_words, activation='softmax')
])
model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=30)
model.save('asl_words_model.h5')
