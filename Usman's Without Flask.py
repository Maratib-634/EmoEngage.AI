import cv2
from deepface import DeepFace
import time
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_data = {'time': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
start_time = time.time()
time_threshold = 20  # Total time threshold (in seconds)
emotion_interval = 5  # Time interval to capture emotions (in seconds)
num_emotions = time_threshold // emotion_interval  # Number of emotions to capture

def calculate_attention_score():
    # Initialize total weight
    total_weight = 0

    # Iterate over emotion data
    for emotion, weight_list in emotion_data.items():
        if emotion != 'time':
            # Sum up the total weight for each emotion
            total_weight += sum(weight_list)

    return total_weight

def detect_emotion():
    cap = cv2.VideoCapture(0)
    emotion_count = 0  # Counter for captured emotions
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
                print(f"Detected emotion: {emotion}")  # Track detected emotion
                attention_weight = {
                    'angry': 0.5,
                    'disgust': 0.25,
                    'fear': 0.25,  # Lower weight for negative emotions
                    'happy': 0.75,
                    'sad': 0.35,  # Even lower weight for strong negative emotions
                    'surprise': 1.0,
                    'neutral': 0.15,
                }.get(emotion, 0)  # Default weight for unknown emotions

                # Update emotion data with attention score
                current_time = time.time() - start_time  # Current time
                emotion_data['time'].append(current_time)  # Update time array
                for key in emotion_data.keys():
                    if key != 'time':
                        if key == emotion:
                            emotion_data[key].append(attention_weight)
                        else:
                            emotion_data[key].append(0)

                emotion_count += 1

        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if all emotions have been captured
        if emotion_count >= num_emotions:
            total_weight = calculate_attention_score()
            print(f"Total Emotion Weight: {total_weight}")
            if total_weight > 1.5:
                print("Engaged")
            else:
                print("Not Engaged")
            break  # End the loop after capturing all emotions

        # Wait for the emotion interval
        time.sleep(emotion_interval)

    cap.release()
    cv2.destroyAllWindows()


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
    while True:
        detect_emotion()
        plot_emotion_graph(emotion_data)
        #user_input = input("Press 'q' to quit, or any other key to continue: ")
        #if user_input.lower() == 'q':
            

