from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_data = {'time': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
start_time = time.time()
time_threshold = 20  # Total time threshold (in seconds)
emotion_interval = 5  # Time interval to capture emotions (in seconds)
num_emotions = time_threshold // emotion_interval  # Number of emotions to capture

def calculate_attention_score():
    total_weight = 0
    for emotion, weight_list in emotion_data.items():
        if emotion != 'time':
            total_weight += sum(weight_list)
    return total_weight

def detect_emotion():
    cap = cv2.VideoCapture(0)
    emotion_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            # Optionally, try to re-initialize the capture:
            cap.release()
            cap = cv2.VideoCapture(0)
            break  # Exit the loop if capture fails


        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            for result in results:
                emotion = result['dominant_emotion']
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
                emotion_data['time'].append(current_time)
                for key in emotion_data.keys():
                    if key != 'time':
                        if key == emotion:
                            emotion_data[key].append(attention_weight)
                        else:
                            emotion_data[key].append(0)

                emotion_count += 1
                return emotion

        if emotion_count >= num_emotions:
            total_weight = calculate_attention_score()
            engagement_status = "Engaged" if total_weight > 1.5 else "Not Engaged"
            return engagement_status

        time.sleep(emotion_interval)

def plot_emotion_graph():
    if not emotion_data['time']:
        print("No data available for plotting")
        return
    
    total_emotion_weights = []
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
    plt.savefig('static/graph.png')
    plt.close()

@app.route('/')
def index():
    plot_emotion_graph()
    emotion = detect_emotion()
    return render_template('index.html', emotion=emotion)

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # Read a frame from the camera
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_engagement_status')
def get_engagement_status():
    total_weight = calculate_attention_score()
    engagement_status = "Engaged" if total_weight > 1.5 else "Not Engaged"
    return engagement_status

if __name__ == '__main__':
    app.run(debug=True)
