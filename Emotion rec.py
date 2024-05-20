import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf

# Load the pre-trained facial expression recognition model
model = tf.keras.models.load_model('C://Users//Hp//Downloads//VGG19.h5', compile=False) # Update with the path to your model file
emotion_labels = ['happy', 'neutral', 'surprise']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # Resize the face image to match the model's expected sizing
        roi_color_resized = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize the pixel values to be between 0 and 1
        roi_color_resized = roi_color_resized / 255.0

        # Reshape the image to (1, 224, 224, 3) to match the model's expected sizing
        roi_color_resized = np.expand_dims(roi_color_resized, axis=0)

        # Make a prediction on the emotion
        prediction = model.predict(roi_color_resized)

        # Get the predicted emotion label
        predicted_emotion = emotion_labels[np.argmax(prediction)]

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
