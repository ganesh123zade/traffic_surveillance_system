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
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['ALERTS_FILE'] = 'alerts.json'
app.config['PREVIOUS_ACCIDENTS_FILE'] = 'previous_accidents.json'
app.config['PREVIOUS_ACCIDENTS_FOLDER'] = 'static/accidents'
app.config['GRAPH_FOLDER'] = 'static/graphs'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure directories exist
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

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(100), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility functions
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
    sender_email = os.getenv('SENDER_EMAIL')
    receiver_email = os.getenv('RECEIVER_EMAIL')
    password = os.getenv('EMAIL_PASSWORD')

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
            recent_alerts.sort(key=lambda x: x['time'], reverse=True)
            f.seek(0)
            f.truncate()
            json.dump(recent_alerts, f)

        with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r+') as f:
            previous_accidents = json.load(f)
            previous_accidents.extend(old_alerts)
            previous_accidents.sort(key=lambda x: x['time'], reverse=True)
            f.seek(0)
            f.truncate()
            json.dump(previous_accidents, f)
    except Exception as e:
        print(f"Failed to move old alerts: {e}")

def detect_accidents(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction, confidence = predict_frame(frame)
        if prediction == 1 and confidence >= 0.9:  # Assuming 1 corresponds to accident class
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

# Routes
@app.route('/')
@login_required
def home():
    with open(app.config['ALERTS_FILE'], 'r') as f:
        alerts = json.load(f)
    with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r') as f:
        previous_accidents = json.load(f)
    return render_template('home.html', alerts=alerts, previous_accidents=previous_accidents)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.is_admin:
                return redirect(url_for('cctv_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            token = secrets.token_hex(16)
            user.reset_token = token
            db.session.commit()
            
            reset_url = url_for('reset_password', token=token, _external=True)
            body = f'Click the link to reset your password: {reset_url}'
            send_email_alert({'photo': '', 'location': '', 'time': time.time(), 'email_body': body})  # Simplified email sending
            
            return 'Reset link sent to your email'
        return 'Email not found'
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user:
        return 'Invalid or expired token'
    
    if request.method == 'POST':
        new_password = request.form['password']
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        user.password = hashed_password
        user.reset_token = None
        db.session.commit()
        return redirect(url_for('login'))
    
    return render_template('reset_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/stream')
@login_required
def stream():
    return render_template('stream.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        video = request.files['video']
        if video:
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(video_path)
            threading.Thread(target=detect_accidents, args=(video_path,)).start()
            return redirect(url_for('home'))
    return render_template('upload.html')

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    return send_from_directory(directory='static/accidents', path=filename)

@app.route('/cctv_dashboard')
@login_required
def cctv_dashboard():
    if not current_user.is_admin:
        return redirect(url_for('home'))
    return render_template('cctv_dashboard.html')

@app.route('/view_graph')
@login_required
def view_graph():
    with open(app.config['PREVIOUS_ACCIDENTS_FILE'], 'r') as f:
        previous_accidents = json.load(f)
    
    times = [datetime.fromtimestamp(alert['time']) for alert in previous_accidents]
    if times:
        hours = [time.hour for time in times]
        hours_count = {hour: hours.count(hour) for hour in range(24)}
        
        plt.bar(hours_count.keys(), hours_count.values())
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Accidents')
        plt.title('Accidents by Hour')
        plt.grid(True)
        graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'accidents_by_hour.png')
        plt.savefig(graph_path)
        plt.close()
        
        return send_from_directory(directory=app.config['GRAPH_FOLDER'], path='accidents_by_hour.png')
    return 'No accident data available'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
