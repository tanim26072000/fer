import cv2
import numpy as np
from keras.models import load_model

# Load the trained model for facial emotion detection
model = load_model('D:/ml-coursera/fer-2/fem.h5')

# List of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # 0 represents the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    # Create a copy of the frame for displaying
    display_frame = cv2.resize(frame, (800, 600))

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Resize the ROI to match the model's input shape (48x48 pixels)
        input_frame = cv2.resize(face_roi, (224, 224))
        input_frame = np.expand_dims(
            input_frame, axis=-1)  # Add channel dimension
        input_frame = input_frame.astype('float32') / 255.0
        input_frame = np.repeat(input_frame, 3, axis=-1)

        # Make a prediction for emotion
        predictions = model.predict(np.expand_dims(input_frame, axis=0))
        emotion_label_index = np.argmax(predictions)
        emotion_name = emotion_labels[emotion_label_index]

        # Draw a rectangle around the detected face on the display frame
        cv2.rectangle(display_frame, (x+100, y+100),
                      (x + w+100, y + h+100), (255, 0, 0), 2)

        # Display the detected emotion under the face bounding box
        cv2.putText(display_frame, emotion_name, (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame with face detections and emotions
    cv2.imshow('Face Emotion Detection', display_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
