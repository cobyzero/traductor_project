import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
import mediapipe as mp

app = Flask(__name__)

# Cargar el modelo Hyper (CNN+LSTM)
model = tf.keras.models.load_model('Bounding_Model.h5')
labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']  # Actualiza con tus clases

# MediaPipe manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Secuencia de landmarks para LSTM
sequence = []

def gen():
    global sequence
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar landmarks
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (250, 250))
        hand_img = hand_img.reshape(1, 250, 250, 1) / 255.0

        prediction_text = "..."

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            sequence.append(coords)

            # Dibujar la mano
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if len(sequence) == 30:
                input_seq = np.expand_dims(sequence, axis=0)  # (1, 30, 21, 3)
                sequence = []

                # Concatenar CNN + LSTM outputs
                cnn_out = model.get_layer('cnn_output')(hand_img)
                lstm_out = model.get_layer('lstm_output')(input_seq)
                combined = tf.concat([cnn_out, lstm_out], axis=1)

                pred = model.predict(combined)
                prediction_text = labels[np.argmax(pred)]

        # Mostrar texto en pantalla
        cv2.putText(frame, prediction_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convertir a JPEG para el navegador
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
