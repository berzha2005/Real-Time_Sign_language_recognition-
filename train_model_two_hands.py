import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
CLASSES = [
    "A", "B", "C", "D", "E", "Hello", "ThankYou", "Please",
    "Beautiful", "Afternoon", "Good", "Morning", "Night" , "I am"
]
FRAMES = 30
FEATURES = 126

X, y = [], []

for label, cls in enumerate(CLASSES):
    for file in os.listdir(os.path.join(DATA_DIR, cls)):
        data = np.load(os.path.join(DATA_DIR, cls, file))
        if data.shape == (FRAMES, FEATURES):
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FRAMES, FEATURES)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))
model.save("sign_model_two_hands.keras")
