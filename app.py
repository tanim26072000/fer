import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

model = load_model('facial_emotion_model.h5')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']


def main():

    st.title("Facial Emotion Recognition App")
    st.write("Use the live camera to detect emotions")
    opencv_placeholder = st.empty()

    # Open a camera capture object
    # 0 indicates the default camera (you can change this value if you have multiple cameras)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break
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
            input_frame = cv2.resize(face_roi, (48, 48))
            input_frame = np.expand_dims(
                input_frame, axis=-1)  # Add channel dimension
            input_frame = input_frame.astype('float32') / 255.0
            input_frame = np.repeat(input_frame, 3, axis=-1)

        # Make a prediction for emotion
            predictions = model.predict(np.expand_dims(input_frame, axis=0))
            emotion_label_index = np.argmax(predictions)
            emotion_name = emotion_labels[emotion_label_index]

        # Draw a rectangle around the detected face on the display frame
            cv2.rectangle(display_frame, (x+100, y+50),
                          (x + w+100, y + h+50), (255, 0, 0), 2)

        # Display the detected emotion under the face bounding box
            cv2.putText(display_frame, emotion_name, (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame with face detections and emotions
        cv2.imshow('Face Emotion Detection', display_frame)

        # Display the OpenCV frame
        opencv_placeholder.image(
            display_frame, channels="BGR", use_column_width=True)

        # Press the "Q" key to exit the live camera feed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Release the camera capture object
    cv2.destroyAllWindows()  # Close any OpenCV windows


if __name__ == "__main__":
    main()
