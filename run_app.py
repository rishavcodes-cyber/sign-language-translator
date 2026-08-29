import pyttsx3
from collections import deque
import cv2
import mediapipe as mp
import pickle
import numpy as np

with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

recent_predictions = deque(maxlen=15)
last_spoken = None
built_word = ""

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        row = []
        for lm in hand_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        prediction = model.predict([row])[0]
        confidence = max(model.predict_proba([row])[0])
        prediction_text = f"{prediction} ({confidence*100:.0f}%)"

        recent_predictions.append(prediction)
        if recent_predictions.count(prediction) > 10:
            if prediction != last_spoken:
                engine.say(prediction)
                engine.runAndWait()
                last_spoken = prediction
                built_word += prediction
    else:
        last_spoken = None  # allows repeating the same letter after removing hand

    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Word: {built_word}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(frame, "Press 'c' to clear, 'q' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sign Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        built_word = ""

cap.release()
cv2.destroyAllWindows()