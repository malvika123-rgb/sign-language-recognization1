import os
import cv2

# Folder where images will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define your gesture classes by name
# Existing gestures + A-Z letters
gestures = ["hello", "thankyou", "iloveyou", "yes", "no"] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

dataset_size = 100  # images per class

# 🔎 Auto-detect a working webcam
cap = None
for i in range(5):  # Try indexes 0,1,2,3,4
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f" Using camera index {i}")
        break

if cap is None:
    raise RuntimeError(" No working camera found")

# Loop through gestures
for gesture in gestures:
    class_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Skip if dataset already exists
    existing_images = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    if existing_images >= dataset_size:
        print(f" Skipping '{gesture}', already collected ({existing_images}/{dataset_size})")
        continue

    print(f' Collecting data for "{gesture}"')
    print("   Press 'Q' when ready to start capturing (or ESC to quit)...")

    # Wait until user presses Q
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Show: {gesture} | Q=start | ESC=quit', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Start capturing
            break
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            print(" Stopped by user")
            exit()

    # Capture remaining images
    counter = existing_images
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC to stop in the middle
            cap.release()
            cv2.destroyAllWindows()
            print(" Stopped by user")
            exit()

        # Save the image
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f" Finished collecting '{gesture}'")

cap.release()
cv2.destroyAllWindows()
print(" Data collection complete!")
