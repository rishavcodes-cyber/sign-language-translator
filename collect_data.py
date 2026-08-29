import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

LABEL = input("Enter the sign label you are about to record (e.g. A, B): ").strip().upper()
SAMPLES_TO_COLLECT = 200

csv_file = "landmark_data.csv"
file_exists = os.path.isfile(csv_file)

cap = cv2.VideoCapture(0)
count = 0

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header)

    while count < SAMPLES_TO_COLLECT:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = [LABEL]
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            writer.writerow(row)
            count += 1

        cv2.putText(frame, f"{LABEL}: {count}/{SAMPLES_TO_COLLECT}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} samples for '{LABEL}' to {csv_file}")