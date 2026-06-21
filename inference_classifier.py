import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
from googletrans import Translator
from gtts import gTTS
import os
import pygame  # sound player

# -------------------------
# LOAD MODEL
# -------------------------
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels = model_dict['labels']

# Separate word gestures and alphabet gestures
word_gestures = [g for g in labels if not g.isalpha()]
alphabet_gestures = [g for g in labels if g.isalpha()]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# -------------------------
# WEBCAM
# -------------------------
cap = None
for i in range(5):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"✅ Using camera index {i}")
        break
if cap is None:
    raise RuntimeError("❌ No camera found")

# -------------------------
# TEXT-TO-SPEECH
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

translator = Translator()
target_lang = None  # None = English, "hi" = Hindi, "kn" = Kannada

# -------------------------
# BUFFERS & STATE
# -------------------------
buffer_size = 10
pred_buffer = deque(maxlen=buffer_size)
current_gesture = ''
locked_gesture = None
sentence_buffer = []
mode = "gesture"

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # =====================================================
    # 🔵 MAKE WINDOW FULLSCREEN (IMPORTANT FOR PRESENTATION)
    # =====================================================
    cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Sign Language Recognition",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # ------------------------------------------------------
    # HAND LANDMARK DETECTION
    # ------------------------------------------------------
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        pred = model.predict([landmarks])
        pred_buffer.append(int(pred[0]))

        if len(pred_buffer) == buffer_size:
            counts = np.bincount(pred_buffer)
            pred_index = np.argmax(counts)
            current_gesture = labels[pred_index] if pred_index < len(labels) else "Unknown"

        # Draw bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(int(min(x_coords) * W) - 20, 0)
        y1 = max(int(min(y_coords) * H) - 20, 0)
        x2 = min(int(max(x_coords) * W) + 20, W)
        y2 = min(int(max(y_coords) * H) + 20, H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # -------------------------
    # DISPLAY UI TEXT
    # -------------------------
    display_gesture = locked_gesture if locked_gesture else current_gesture
    cv2.putText(frame, f"Gesture: {display_gesture}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)

    if mode == "sentence":
        display_text = ''.join(sentence_buffer)
        cv2.putText(frame, f"Sentence: {display_text}", (30, H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)

    # -------------------------
    # KEY EVENTS
    # -------------------------
    key = cv2.waitKey(1) & 0xFF

    if key in [27, ord('q')]:  # Quit
        break

    elif key == ord('s'):  # sentence mode
        mode = "sentence"
        print("🟦 Switched to Sentence Mode")

    elif key == ord('g'):  # gesture mode
        mode = "gesture"
        print("🟩 Switched to Gesture Mode")

    elif key == ord('d'):  # lock gesture
        if current_gesture != "Unknown":
            locked_gesture = current_gesture
            print(f"🔒 Locked gesture: {locked_gesture}")

    elif key == ord('a'):  # accept locked gesture
        if locked_gesture:
            if mode == "gesture" and locked_gesture in word_gestures:
                print(f"🔊 Speaking: {locked_gesture}")
                engine.say(locked_gesture)
                engine.runAndWait()

            elif mode == "sentence" and locked_gesture in alphabet_gestures:
                sentence_buffer.append(locked_gesture)

            locked_gesture = None

    elif key == 13:  # ENTER → speak sentence
        if mode == "sentence" and sentence_buffer:
            sentence = ''.join(sentence_buffer)
            print(f"🗣 Speaking (English): {sentence}")
            engine.say(sentence)
            engine.runAndWait()

            if target_lang:
                translation = translator.translate(sentence, dest=target_lang).text
                print(f"🌐 Translated ({target_lang}): {translation}")

                tts = gTTS(text=translation, lang=target_lang)
                filename = "temp_translation.mp3"
                tts.save(filename)

                # Safe playback using pygame
                try:
                    pygame.mixer.quit()
                    pygame.mixer.init()
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                finally:
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                    if os.path.exists(filename):
                        os.remove(filename)

    elif key == 8:  # BACKSPACE
        if mode == "sentence" and sentence_buffer:
            sentence_buffer.pop()

    elif key == 32:  # SPACE
        if mode == "sentence":
            sentence_buffer.append(" ")

    elif key == ord('c'):  # clear
        sentence_buffer = []

    elif key == ord('0'):
        target_lang = None
        print("🌍 Language: English")

    elif key == ord('1'):
        target_lang = "hi"
        print("🌍 Language: Hindi")

    elif key == ord('2'):
        target_lang = "kn"
        print("🌍 Language: Kannada")

cap.release()
cv2.destroyAllWindows()


