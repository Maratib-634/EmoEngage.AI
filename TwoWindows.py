
from flask import Flask, render_template, Response, send_from_directory
import cv2
from deepface import DeepFace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

app = Flask(__name__)
STATIC_FOLDER = r'P:\Working Directories\Flask\EmoEngage\static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
url = 'http://192.168.18.19:4747'

emotion_data = {'time': [], 'anger': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
start_time = time.time()

def detect_emotion():
    cap = cv2.VideoCapture(0)
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
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if result:
                emotion = result[0]['emotion']
                current_time = time.time() - start_time
                emotion_data['time'].append(current_time)
                for emo in emotion_data.keys():
                    if emo != 'time':
                        emotion_data[emo].append(emotion.get(emo, 0))  # Ensure emotion exists in the result

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Unable to encode frame")
            break
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_graph')
def emotion_graph():
    plt.switch_backend('Agg')  # Switch backend to Agg
    plt.figure(figsize=(10, 6))
    for emo in emotion_data.keys():
        if emo != 'time':
            plt.plot(emotion_data['time'], emotion_data[emo], label=emo)
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion')
    plt.title('Emotion over Time')
    plt.legend()
    plt.grid(True)
    graph_path = os.path.join(app.config['STATIC_FOLDER'], 'emotion_graph.png')
    plt.savefig(graph_path)  # Save the graph as a PNG file
    plt.close()  # Close the plot to release resources
    return send_from_directory(app.config['STATIC_FOLDER'], 'emotion_graph.png')

if __name__ == '__main__':
    app.run(debug=True)
