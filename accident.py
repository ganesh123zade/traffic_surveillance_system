import cv2
import tensorflow as tf
import os
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import sys

if os.name == 'nt':  # Check if the OS is Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Step 1: Extract frames from the video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite(os.path.join(output_folder, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1

# Step 2: Open the video frames in a window
def display_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = vidcap.read()

        if not success:
            break

        cv2.imshow('Video Playback', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidcap.release()
    cv2.destroyAllWindows()

# Step 3: Prepare data for training
def prepare_data(train_data_dir, validation_data_dir):
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
    
    return train_generator, validation_generator

# Step 4: Define and train the model
def create_and_train_model(train_generator, validation_generator):
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
    
    return model

# Step 5: Simulate GPS data
def get_current_location():
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    return latitude, longitude

# Step 6: Send alert with location and screenshot
def send_alert(location, screenshot_path):
    sender = 'prinshis673@gmail.com'
    receiver = 'ganesh123zade@gmail.com'
    subject = 'Accident Alert'
    body = f'An accident has been detected at location: {location}'

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with open(screenshot_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(screenshot_path)}')
        msg.attach(part)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:  # Using SMTP_SSL with port 465
        server.login(sender, 'fuhx opow qbqq ufkf')  # Replace with your app-specific password
        server.sendmail(sender, receiver, msg.as_string())

# Step 7: Detect accident and send alert
def detect_accident(frame, model):
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0
    frame = frame.reshape(1, 150, 150, 3)
    prediction = model.predict(frame)
    return prediction[0][0] > 0.5

def main():
    # Adjust paths as necessary
    video_path = 'accident.mp4'
    output_frames_folder = 'output_frames_folder'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(script_dir, 'train')
    validation_data_dir = os.path.join(script_dir, 'validation')
    screenshot_path = os.path.join(output_frames_folder, 'screenshot.jpg')

    # Step 1: Extract frames from the video
    extract_frames(video_path, output_frames_folder)
  
    # Step 2: Display the video (optional)
    display_video(video_path)

    # Step 3: Prepare data for training
    train_generator, validation_generator = prepare_data(train_data_dir, validation_data_dir)

    

    # Step 4: Define and train the model
    model = create_and_train_model(train_generator, validation_generator)

    # Step 7: Process video to detect accidents
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    while success:
        if detect_accident(frame, model):
            location = get_current_location()
            cv2.imwrite(screenshot_path, frame)
            send_alert(location, screenshot_path)
            break
        success, frame = vidcap.read()

    vidcap.release()

if __name__ == "__main__":
    main()
