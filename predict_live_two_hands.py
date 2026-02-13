import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

MODEL_PATH = "sign_model_two_hands.keras"
CLASSES = [
    "A", "B", "C", "D", "E", "Hello", "ThankYou", "Please",
    "Beautiful", "Afternoon", "Good", "Morning", "Night", "I am"
]
FRAMES = 30
CONFIDENCE_THRESHOLD = 0.75

model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
frames_collected = []
sentence = ""
last_word = ""

def extract_two_hands(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts = []
            for p in lm.landmark:
                pts.extend([p.x, p.y, p.z])
            if hand.classification[0].label == "Left":
                left = np.array(pts)
            else:
                right = np.array(pts)
    return np.concatenate([left, right])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        frame_data = extract_two_hands(results)
        frames_collected.append(frame_data)

        if len(frames_collected) == FRAMES:
            input_data = np.expand_dims(frames_collected, axis=0)
            preds = model.predict(input_data, verbose=0)
            conf = np.max(preds)
            idx = np.argmax(preds)

            if conf > CONFIDENCE_THRESHOLD:
                word = CLASSES[idx]
                if word != last_word:
                    sentence += word + " "
                    last_word = word

            frames_collected = []

    else:
        frames_collected = []

    cv2.putText(frame, f"Sentence: {sentence}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Two Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
