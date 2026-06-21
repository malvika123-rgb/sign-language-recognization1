import os
import cv2
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp

DATA_DIR = './data'
GESTURES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

print("Detected gesture classes:", GESTURES)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data = []
labels = []

IMG_SIZE = (64, 64)  # not used here, just for reference

print(" Extracting landmarks from images...")

for idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for img_name in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            data.append(landmarks)
            labels.append(idx)
        else:
            continue  # Skip images where hand was not detected

data = np.array(data)
labels = np.array(labels)
print(f" Extracted landmarks from {len(data)} images")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100)
print(" Training RandomForest classifier...")
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(f"Accuracy: {score*100:.2f}%")

# Save model and gesture labels
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': GESTURES}, f)

print("Model saved as model.p")
