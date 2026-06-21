# sign-language-recognization1
# Real-Time Sign Language Recognition (AI/ML)

A real-time Sign Language Recognition system that detects hand gestures via webcam and converts them into text and speech, with multilingual translation support.

## Features

- Real-time hand gesture detection using **MediaPipe** hand landmarks
- Gesture classification using a **Random Forest** model (scikit-learn)
- Recognizes custom word gestures (hello, thank you, yes, no, etc.) and a custom A–Z hand sign alphabet
- **Sentence mode**: spell out words letter-by-letter into a sentence
- **Text-to-speech** output (pyttsx3 / gTTS)
- **Multilingual translation** (English, Hindi, Kannada) before speaking
- Simple **login/register** screen (Tkinter) to launch the recognizer

## Tech Stack

Python · OpenCV · MediaPipe · scikit-learn · pyttsx3 · gTTS · googletrans · pygame · Tkinter

## Gesture Reference

All gestures used in this project — both the alphabet (A–Z) and the words (hello, thank you, yes, no, I love you) — are **custom hand signs**, collected and trained specifically for this project (not standard ASL).you can refer ASL images for reference and make custom hand signs 

## Setup

1. Clone this repository.
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the app:
```bash
   python login.py
```
   Register a username/password, log in, and the gesture recognition window will launch automatically.

(Optional — only needed if you want to retrain on your own gestures)
```bash
python collect_imgs.py        # collect webcam images per gesture
python create_dataset.py      # build landmark dataset
python train_classifier.py    # train and save model.p
```

## Controls (during recognition)

| Key | Action |
|-----|--------|
| `g` | Switch to Gesture (word) mode |
| `s` | Switch to Sentence (spelling) mode |
| `d` | Lock the currently detected gesture |
| `a` | Accept the locked gesture (speak word / add letter) |
| `Enter` | Speak the full sentence (with translation) |
| `Backspace` | Delete last letter in sentence |
| `Space` | Add a space in sentence |
| `c` | Clear sentence |
| `0` / `1` / `2` | Set output language: English / Hindi / Kannada |
| `q` / `Esc` | Quit |

## Notes

- Training images (`data/`) are not included since the model (`model.p`) is already trained and ready to use. Run `collect_imgs.py` if you want to build your own dataset.
- `users.json` (login credentials) is excluded for privacy and is created automatically on first registration.


Malvika Naik
