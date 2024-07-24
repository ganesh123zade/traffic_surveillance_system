# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('accident_detection_model.h5')
# import cv2
# import numpy as np

# def predict_frame(frame):
#     # Preprocess the frame
#     img = cv2.resize(frame, (224, 224))
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0  # Rescale to [0,1]

#     # Predict probabilities
#     predictions = model.predict(img)
    
#     # Get class with highest probability
#     predicted_class = np.argmax(predictions)
    
#     return predicted_class, predictions[0][predicted_class]
# def detect_accidents(video_path):
#     cap = cv2.VideoCapture(video_path)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Perform prediction on the frame
#         prediction, confidence = predict_frame(frame)
        
#         # Decide if there is an accident based on the prediction
#         if prediction == 1:  # Assuming 1 corresponds to accident class
#             # Perform actions for accident detected
#             print("Accident detected with confidence:", confidence)
#             # For example, save the frame, alert authorities, etc.
        
#         # Display the frame (optional)
#         cv2.imshow('Frame', frame)
        
#         # Exit key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_path = 'accident4.mp4'  # Replace with your video file path
# detect_accidents(video_path)

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import os
import uuid  # For generating unique names for saved frames

# Load the trained model
model = load_model('accident_detection_model.h5')

def create_output_directory(video_path, base_dir='./data_1'):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_directory = os.path.join(base_dir, video_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory

def predict_frame(frame):
    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Rescale to [0,1]

    # Predict probabilities
    predictions = model.predict(img)
    
    # Get class with highest probability
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

def detect_accidents(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform prediction on the frame
        prediction, confidence = predict_frame(frame)
        
        # Decide if there is an accident based on the prediction
        if prediction == 1 and confidence >= 0.9:  # Adjust confidence threshold as needed
            # Generate a unique filename
            frame_count += 1
            filename = f'accident_frame_{str(uuid.uuid4())[:8]}.jpg'
            output_directory=create_output_directory(video_path)
            save_path = os.path.join(output_directory, filename)
            
            # Save the frame
            cv2.imwrite(save_path, frame)
            print(f"Saved high confidence accident frame to {filename}")
        
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'accident.mp4'  # Replace with your video file path
detect_accidents(video_path)
