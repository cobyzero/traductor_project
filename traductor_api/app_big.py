from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model_asl_rgb.h5')
labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

cap = cv2.VideoCapture(0)
history = []

def preprocess(frame):
    h, w, _ = frame.shape
    min_dim = min(h, w)
    x = (w - min_dim) // 2
    y = (h - min_dim) // 2
    cropped = frame[y:y+min_dim, x:x+min_dim]
    resized = cv2.resize(cropped, (200, 200))
    normalized = resized / 255.0
    return normalized.reshape(1, 200, 200, 3)

def generate_frames():
    global history
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess(frame)
        pred = model.predict(img, verbose=0)
        letter = labels[np.argmax(pred)]
        history.append(letter)

        cv2.putText(
            frame, f'Letra: {letter}', (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html', history=' '.join(history[-30:]))

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
