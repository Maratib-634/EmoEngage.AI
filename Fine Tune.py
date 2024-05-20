from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

# Load the pre-trained facial expression recognition model
model = load_model('C:\\Users\\Hp\\Downloads\\saved_model.h5')  # Update with the path to your model file
emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'surprise']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess an image
def preprocess_image(img_array):
    img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face in the frame
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) - the face
        roi_color = frame[y:y + h, x:x + w]

        # Resize and preprocess the face image
        preprocessed_image = preprocess_image(roi_color)

        # Make a prediction on the emotion
        prediction = model.predict(preprocessed_image)

        # Get the predicted emotion label
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


