from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from deepface import DeepFace
import time
import numpy as np
import threading
import os
from queue import Queue

app = Flask(__name__)

# Define STATIC_FOLDER to store static files
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Initialize OpenCV cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize global variables for emotion data and start time
emotion_data_queue = Queue()
start_time = time.time()

# Function to calculate attention score
def calculate_attention_score():
    total_weight = 0
    while not emotion_data_queue.empty():
        emotion_data_point = emotion_data_queue.get()
        total_weight += sum(emotion_data_point.values())
    return total_weight

def detect_emotion(cap):
    emotion_count = 0
    num_emotions = 100  # Define the number of emotions to capture
    emotion_interval = 1  # Define the interval between capturing emotions (in seconds)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            for result in results:
                emotion = result['dominant_emotion']
                print(f"Detected emotion: {emotion}")
                attention_weight = {
                    'angry': 0.5,
                    'disgust': 0.25,
                    'fear': 0.25,
                    'happy': 0.75,
                    'sad': 0.35,
                    'surprise': 1.0,
                    'neutral': 0.15,
                }.get(emotion, 0)
                current_time = time.time() - start_time
                emotion_data_point = {'time': current_time}
                for key in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                    emotion_data_point[key] = attention_weight if key == emotion else 0
                emotion_data_queue.put(emotion_data_point)
                emotion_count += 1

        if emotion_count >= num_emotions:
            total_weight = calculate_attention_score()
            print(f"Total Emotion Weight: {total_weight}")
            if total_weight > 1.5:
                print("Engaged")
            else:
                print("Not Engaged")
            break
        time.sleep(emotion_interval)

    print("Emotion data collected:")
    print(emotion_data_queue)

# Route to display index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get attention score
@app.route('/get_attention_score')
def get_attention_score():
    attention_score= calculate_attention_score()
    return jsonify({'attention_score': attention_score})

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        detect_emotion(cap)
        cap.release()
        cv2.destroyAllWindows()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to display emotion graph
@app.route('/emotion_graph')
def emotion_graph():
    plot_emotion_graph()
    return send_from_directory(app.config['STATIC_FOLDER'], 'emotion_graph.png')

def plot_emotion_graph(emotion_data):
    if not emotion_data['time']:
        print("No data available for plotting")
        return
    
    total_emotion_weights = []  # List to store total emotion weight at each time point
    for i, time_point in enumerate(emotion_data['time']):
        total_weight = sum([emotion_data[emotion][i] for emotion in emotion_data.keys() if emotion != 'time'])
        total_emotion_weights.append(total_weight)

    plt.figure(figsize=(10, 6))
    plt.plot(emotion_data['time'], total_emotion_weights, label='Total Emotion Weight', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Emotion Weight')
    plt.title('Total Emotion Weight over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Start a separate thread for emotion detection
    t = threading.Thread(target=detect_emotion, args=(cv2.VideoCapture(0),))
    t.start()

    # Run the Flask app
    app.run(debug=True)