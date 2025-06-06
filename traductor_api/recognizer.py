import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

def recognize_hand_gesture(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return "No se detectó una mano"

        # Por ahora solo retorna "A" como prueba
        return "A"
