import cv2
import os
import numpy as np
import mediapipe as mp

# ---------------- CONFIG ----------------
DATA_DIR = "data"

CLASSES = [
    "A", "B", "C", "D", "E", "F" ,  "Hello", "ThankYou", "Please","Beautiful", "Afternoon", "Good", "Morning", "Night", "I am"
]

FRAMES_PER_SAMPLE = 30
FEATURES_PER_HAND = 63
# ---------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

current_class_index = 0
recording = False
frames_collected = []

print("n = next | p = previous | r = record | q = quit")

def get_two_hand_landmarks(results):
    left_hand = np.zeros(FEATURES_PER_HAND)
    right_hand = np.zeros(FEATURES_PER_HAND)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if handedness.classification[0].label == "Left":
                left_hand = np.array(landmarks)
            else:
                right_hand = np.array(landmarks)

    return np.concatenate([left_hand, right_hand])
def draw_ui(frame, current_word, recording, frames_count, total_frames):
    h, w, _ = frame.shape

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    status_text = "RECORDING" if recording else "IDLE"
    status_color = (0, 0, 255) if recording else (0, 200, 0)

    cv2.putText(
        frame, f"WORD: {current_word}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 200, 0),
        2
    )

    cv2.putText(
        frame, f"STATUS: {status_text}",
        (w - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    # Progress bar
    bar_x, bar_y = 50, h - 80
    bar_width, bar_height = w - 100, 20

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (80, 80, 80),
        2
    )

    if recording:
        progress = int((frames_count / total_frames) * bar_width)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + progress, bar_y + bar_height),
            (0, 0, 255),
            -1
        )

    cv2.putText(
        frame,
        f"{frames_count}/{total_frames}",
        (bar_x + bar_width - 120, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "n: next   p: prev   r: record   q: quit",
        (50, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS
            )

        if recording:
            frame_data = get_two_hand_landmarks(results)
            frames_collected.append(frame_data)

            if len(frames_collected) == FRAMES_PER_SAMPLE:
                current_class = CLASSES[current_class_index]
                class_path = os.path.join(DATA_DIR, current_class)
                os.makedirs(class_path, exist_ok=True)

                count = len(os.listdir(class_path))
                file_path = os.path.join(
                    class_path, f"sample_{count + 1}.npy"
                )

                np.save(file_path, np.array(frames_collected))
                print("Saved:", file_path)

                frames_collected = []
                recording = False

    # ✅ NEW UI (only this)
    draw_ui(
        frame,
        CLASSES[current_class_index],
        recording,
        len(frames_collected),
        FRAMES_PER_SAMPLE
    )

    cv2.imshow("Sign Language Data Collection", frame)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        current_class_index = (current_class_index + 1) % len(CLASSES)
        print("Selected:", CLASSES[current_class_index])

    elif key == ord('p'):
        current_class_index = (current_class_index - 1) % len(CLASSES)
        print("Selected:", CLASSES[current_class_index])

    elif key == ord('r'):
        recording = True
        frames_collected = []
        print("Recording:", CLASSES[current_class_index])

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
