from flask import Flask, render_template, jsonify, request, send_from_directory, Response, stream_with_context
import os
import smtplib
import uuid
import json
import cv2
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['ALERTS_FILE'] = 'alerts.json'
app.config['PREVIOUS_ACCIDENTS_FILE'] = 'previous_accidents.json'
app.config['PREVIOUS_ACCIDENTS_FOLDER'] = 'static/accidents'
app.config['GRAPH_FOLDER'] = 'static/graphs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREVIOUS_ACCIDENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

# Initialize JSON files
if not os.path.exists(app.config['ALERTS_FILE']):
    with open(app.config['ALERTS_FILE'], 'w') as f:
        json.dump([], f)

if not os.path.exists(app.config['PREVIOUS_ACCIDENTS_FILE']):
    with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'w') as f:
        json.dump([], f)

# Load the trained model
model = load_model('accident_detection_model.h5')

def create_output_directory(video_path):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_directory = os.path.join('static/accidents', video_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def predict_frame(frame):
    img = preprocess_frame(frame)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence

def send_email_alert(alert):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = os.getenv('abhishekdubalgonde7846@gmail.com')
    receiver_email = os.getenv('ganesh123zade@gmail.com')
    password = os.getenv('fihm tmhi zhlp gbma')

    subject = "Accident Detected"
    body = f"An accident was detected at {datetime.fromtimestamp(alert['time']).strftime('%Y-%m-%d %H:%M:%S')}.\nLocation: {alert['location']}\n\n"
    google_maps_link = f"https://www.google.com/maps/search/?api=1&query={alert['location'].replace(' ', '+')}"
    body += f"Google Maps Directions: {google_maps_link}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(alert['photo'], 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(alert['photo'])}")
        msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_alert(alert):
    try:
        with open(app.config['ALERTS_FILE'], 'r+') as f:
            data = json.load(f)
            data.append(alert)
            f.seek(0)
            json.dump(data, f)
        update_timestamp()
    except Exception as e:
        print(f"Failed to save alert: {e}")

def update_timestamp():
    timestamp = time.time()
    with open('timestamp.json', 'w') as f:
        json.dump({'timestamp': timestamp}, f)

def move_old_alerts():
    five_minutes_ago = time.time() - 300
    try:
        with open(app.config['ALERTS_FILE'], 'r+') as f:
            alerts = json.load(f)
            recent_alerts = [alert for alert in alerts if alert['time'] >= five_minutes_ago]
            old_alerts = [alert for alert in alerts if alert['time'] < five_minutes_ago]
            recent_alerts.sort(key=lambda x: x['time'], reverse=True)  # Sort recent alerts in descending order
            f.seek(0)
            f.truncate()
            json.dump(recent_alerts, f)

        with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r+') as f:
            previous_accidents = json.load(f)
            previous_accidents.extend(old_alerts)
            previous_accidents.sort(key=lambda x: x['time'], reverse=True)  # Sort previous accidents in descending order
            f.seek(0)
            f.truncate()
            json.dump(previous_accidents, f)
    except Exception as e:
        print(f"Failed to move old alerts: {e}")

def detect_accidents(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction, confidence = predict_frame(frame)
        if prediction == 1 and confidence >= 0.9:  # Assuming 1 corresponds to accident class
            frame_count += 1
            filename = f'accident_frame_{str(uuid.uuid4())[:8]}.jpg'
            output_directory = create_output_directory(video_path)
            save_path = os.path.join(output_directory, filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved high confidence accident frame to {filename}")
            timestamp = datetime.now(timezone.utc).timestamp()
            alert = {
                'time': timestamp,
                'location': '1234 Example St, City, Country',  # Replace with actual location
                'photo': save_path
            }
            save_alert(alert)
            send_email_alert(alert)
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    with open(app.config['ALERTS_FILE'], 'r') as f:
        alerts = json.load(f)
    with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r') as f:
        previous_accidents = json.load(f)
    return render_template('index.html', alerts=alerts, previous_accidents=previous_accidents)

@app.route('/alerts')
def get_alerts():
    with open(app.config['ALERTS_FILE'], 'r') as f:
        alerts = json.load(f)
    return jsonify(alerts)

@app.route('/previous_accidents')
def get_previous_accidents():
    with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r') as f:
        previous_accidents = json.load(f)
    return jsonify(previous_accidents)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video part', 400

    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    threading.Thread(target=detect_accidents, args=(file_path,)).start()
    
    return 'Video uploaded successfully', 200

@app.route('/stream_analysis')
def stream_analysis():
    def generate():
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('video_filename'))
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            prediction, confidence = predict_frame(frame)
            frame_data = {
                "status": "processing",
                "frame": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "data": frame.flatten().tolist()  # Convert frame to list of values
                }
            }
            yield f"data: {json.dumps(frame_data)}\n\n"

            if prediction == 1 and confidence >= 0.9:
                frame_data["status"] = "accident"
                yield f"data: {json.dumps(frame_data)}\n\n"

        cap.release()
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/graph')
def graph_page():
    return render_template('graphical_analysis.html')

@app.route('/cctv_dashboard')
def cctv_dashboard():
    return render_template('cctv_dashboard.html')

@app.route('/generate_graph')
def generate_graph():
    with open(app.config['ALERTS_FILE'], 'r') as f:
        alerts = json.load(f)
    
    dates = [datetime.fromtimestamp(alert['time']).strftime('%Y-%m-%d') for alert in alerts]
    unique_dates = list(set(dates))
    counts = [dates.count(date) for date in unique_dates]
    
    plt.figure(figsize=(10, 5))
    plt.bar(unique_dates, counts, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents per Day')
    plt.xticks(rotation=45)
    
    graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'accidents_per_day.png')
    plt.savefig(graph_path)
    plt.close()

    return send_from_directory(app.config['GRAPH_FOLDER'], 'accidents_per_day.png')

@app.route('/timestamp')
def get_timestamp():
    try:
        with open('timestamp.json', 'r') as f:
            timestamp = json.load(f)
        return jsonify(timestamp)
    except Exception as e:
        print(f"Failed to read timestamp: {e}")
        return jsonify({'timestamp': 0})

if __name__ == '__main__':
    threading.Thread(target=move_old_alerts).start()
    app.run(debug=True , port=5001)
