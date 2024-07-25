import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import threading
import cv2
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

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

def check_for_accident(frame, background_subtractor, min_area=500):
    fg_mask = background_subtractor.apply(frame)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        return True
    return False

def detect_accidents(video_source=0):
    cap = cv2.VideoCapture(video_source)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    accident_detected = False
    accident_count = 0
    max_accident_images = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if check_for_accident(frame, background_subtractor):
            if not accident_detected:
                accident_detected = True
                accident_count = 0

            if accident_count < max_accident_images:
                timestamp = datetime.now(timezone.utc).timestamp()
                formatted_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                photo_path = f'static/accidents/accident_{formatted_time}.jpg'
                cv2.imwrite(photo_path, frame)
                location = '1234 Example St, City, Country'
                alert = {
                    'time': timestamp,
                    'location': location,
                    'photo': photo_path
                }
                save_alert(alert)
                send_email_alert(alert)
                accident_count += 1
        else:
            accident_detected = False

        time.sleep(1)
    cap.release()

def send_email_alert(alert):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = 'your_email@gmail.com'  # Update with your sender email
    receiver_email = 'recipient_email@gmail.com'  # Update with recipient email
    password = 'your_password'  # Update with your email password

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
    five_minutes_ago = time.time() - 300  # 300 seconds = 5 minutes
    try:
        with open(app.config['ALERTS_FILE'], 'r+') as f:
            alerts = json.load(f)
            recent_alerts = [alert for alert in alerts if alert['time'] >= five_minutes_ago]
            old_alerts = [alert for alert in alerts if alert['time'] < five_minutes_ago]
            f.seek(0)
            f.truncate()
            json.dump(recent_alerts, f)

        with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r+') as f:
            previous_accidents = json.load(f)
            previous_accidents.extend(old_alerts)
            f.seek(0)
            f.truncate()
            json.dump(previous_accidents, f)
    except Exception as e:
        print(f"Failed to move old alerts: {e}")

@app.route('/timestamp')
def get_timestamp():
    try:
        with open('timestamp.json', 'r') as f:
            timestamp = json.load(f)
        return jsonify(timestamp)
    except Exception as e:
        print(f"Failed to read timestamp: {e}")
        return jsonify({'timestamp': 0})

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

if __name__ == '__main__':
    threading.Thread(target=move_old_alerts).start()
    app.run(debug=True)
