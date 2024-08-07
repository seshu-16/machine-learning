import cv2
import os
import numpy as np

def load_images_from_folder(base_folder):
    images = []
    labels = []
    
    subfolders = ['Open-Eyes', 'Close-Eyes']
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                if 'closed' in subfolder.lower():
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                print(f"Warning: Unable to read image {img_path}")
    
    return images, labels

base_folder_path = "mrleyedataset"  # Base folder containing the subfolders
images, labels = load_images_from_folder(base_folder_path)

# Ensure some images were loaded
if len(images) == 0:
    raise ValueError("No images were loaded. Check the dataset path and image files.")

# Resize images
images_resized = [cv2.resize(img, (64, 64)) for img in images]

# Normalize images
images_normalized = np.array(images_resized) / 255.0
labels = np.array(labels)

print("Images loaded and preprocessed successfully.")

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(images_normalized, labels, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(images_normalized, labels)
print(f"Test Accuracy: {test_acc}")



import cv2
import numpy as np
from playsound import playsound

def detect_drowsiness(frame, model):
    resized_frame = cv2.resize(frame, (64, 64)) / 255.0
    prediction = model.predict(np.expand_dims(resized_frame, axis=0))
    return prediction

cap = cv2.VideoCapture(0)

drowsiness_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    prediction = detect_drowsiness(frame, model)
    if prediction < 0.5:
        cv2.putText(frame, 'Drowsy', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not drowsiness_detected:
            drowsiness_detected = True
            playsound('alert-sound.mp3')
            playsound('take_a_break.mp3')
            playsound('focus.mp3')
    else:
        cv2.putText(frame, 'Awake', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        drowsiness_detected = False
    
    cv2.imshow('Driver Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
