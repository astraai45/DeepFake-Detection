import streamlit as st
import numpy as np
import keras
import pandas as pd
import os
import cv2
import face_recognition
import tensorflow as tf
from tensorflow.keras.models import load_model

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
SEQ_LENGTH = 20

# Check if dlib is installed
try:
    import dlib
except ImportError:
    # Run the build script to install dlib
    subprocess.run(['./build.sh'], check=True)

# Function to extract face from frames
def crop_face_center(frame):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        face_location = face_locations[0]
    else:
        # If face not detected, use default image
        default_image = cv2.imread('default_image.png')
        face_locations = face_recognition.face_locations(default_image)
        face_location = face_locations[0]
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    return face_image

import io

import os
import cv2
import numpy as np
import tempfile

def load_video(uploaded_file, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    # Save the uploaded file to a temporary location
    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(temp_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / SEQ_LENGTH), 1)
    frames = []
    
    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
        print("Completed extracting frames")
    finally:
        cap.release()
        # Remove the temporary file
        os.remove(temp_file_path)
    
    return np.array(frames)





# Function to extract features from Video Frames
def build_feature_extractor():
    from tensorflow.keras.applications import InceptionV3

    IMG_SIZE = 224
    # Using pretrained InceptionV3 Model
    feature_extractor = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Adding a Preprocessing layer in the Model
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Load the pretrained sequence model
sequence_model = load_model('models/inceptionNet_model.h5')

# Create the feature extractor model
feature_extractor = build_feature_extractor()

# Function to prepare a single video for prediction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, 20,), dtype="bool")
    frame_features = np.zeros(shape=(1, 20, 2048), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(20, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
    return frame_features, frame_mask

# Function to predict authenticity of a video
def predict_video(video_file):
    frames = load_video(video_file)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    fake_probability = probabilities[0] * 100  # Probability of being fake
    return fake_probability

# Streamlit app
def main():
    st.title("DeepFake Detection")
    st.write("Upload a video to predict if it's fake or real.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    if uploaded_file is not None:
        # Display the uploaded video
        st.video(uploaded_file)

        # Predict authenticity if the user clicks the button
        if st.button("Detect"):
            fake_probability = predict_video(uploaded_file)
            result = "Fake" if fake_probability > 50 else "Real"
            st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
