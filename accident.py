import cv2
import tensorflow as tf
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
import smtplib
from email.mime.text import MIMEText

# Step 1: Extract frames from the video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)  # Use the provided video_path
    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite(os.path.join(output_folder, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1

# Call the extract_frames function with the correct video path
extract_frames('path_to_video.mp4', 'output_frames_folder')

# Open the video frames in a window
video_path = 'accident.mp4'  # Provide the correct path to the video
vidcap = cv2.VideoCapture('accident.mp4')

while True:
    success, frame = vidcap.read()

    if not success:
        break

    cv2.imshow('accident', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))


# Step 2: Prepare data for training
train_data_dir = os.path.join(script_dir, 'train')
validation_data_dir = os.path.join(script_dir, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Step 3: Define and train the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=25, validation_data=validation_generator)

# Step 4: Simulate GPS data
def get_current_location():
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    return latitude, longitude

# Step 5: Send alert with location
# Step 5: Send alert with location
def send_alert(location):
    sender = 'prinshis673@gmail.com'
    receiver = 'rohit.panchal.9212@gmail.com'
    subject = 'Accident Alert'
    body = f'An accident has been detected at location: {location}'

    # Encode body using UTF-8 encoding
    encoded_body = body.encode('utf-8')

    msg = MIMEText(encoded_body, _charset='utf-8')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    # Connect to SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, 'vwal fiom ygzs guob')
        server.sendmail(sender, receiver, msg.as_string())




def detect_accident(frame):
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0
    frame = frame.reshape(1, 150, 150, 3)
    prediction = model.predict(frame)
    return prediction[0][0] > 0.5

video_path = 'accident.mp4'
vidcap = cv2.VideoCapture('accident.mp4')
success, frame = vidcap.read()

while success:
    if detect_accident(frame):
        location = get_current_location()
        send_alert(location)
        break
    success, frame = vidcap.read()
