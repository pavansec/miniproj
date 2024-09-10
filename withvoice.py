import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
from PIL import ImageFont, ImageDraw, Image

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture from the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize the labels dictionary (Tamil characters)
labels_dict = {
    0: 'அம்மா', 1: 'அப்பா', 2: 'இ', 3: 'ஈ', 4: 'உ', 5: 'ஊ', 
    6: 'எ', 7: 'ஏ', 8: 'ஐ', 9: 'ஒ', 10: 'ஓ', 11: 'ஔ',
}

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Create a folder to save all audio files
audio_dir = 'audio_files'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Generate and save audio for each Tamil character
for key, tamil_character in labels_dict.items():
    audio_file = os.path.join(audio_dir, f'{key}.mp3')

    # Check if the file already exists and delete it
    if os.path.exists(audio_file):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()  # Stop any playback if necessary
        os.remove(audio_file)

    # Generate Tamil text-to-speech for each character
    tts = gTTS(text=tamil_character, lang='ta')
    tts.save(audio_file)  # Save the audio file

# Load a Tamil font using PIL
tamil_font = ImageFont.truetype("NotoSansTamil-VariableFont_wdth,wght.ttf", 40)  # Specify the path to your .ttf file

# Main loop to capture video frames and predict hand gestures
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the gesture and get the corresponding Tamil character
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Use PIL to render Tamil characters
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL format
        draw = ImageDraw.Draw(frame_pil)
        
        # Draw the Tamil character using PIL
        draw.text((x1, y1 - 50), predicted_character, font=tamil_font, fill=(0, 128, 0))

        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Draw the rectangle using OpenCV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Play the corresponding audio for the predicted Tamil character
        audio_file = os.path.join(audio_dir, f'{int(prediction[0])}.mp3')
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()  # Ensure previous audio is stopped
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

    # Show the video feed with hand tracking and prediction
    cv2.imshow('frame', frame)

    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
