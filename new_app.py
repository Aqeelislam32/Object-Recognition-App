import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import Image

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Function to predict the class and confidence
def predict_currency(image):
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Streamlit App
st.title("Currency Detection App")

# Sidebar options
option = st.sidebar.selectbox("Choose an input method:", ["Webcam", "IP CCTV", "Upload Image"])

if option == "Webcam":
    st.subheader("Live Webcam Prediction")
    start_button = st.button("Start Webcam")

    if start_button:
        camera = cv2.VideoCapture(0)
        st_frame = st.empty()

        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access the webcam.")
                break

            # Show the webcam feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")

            # Preprocess and predict
            processed_frame = preprocess_image(frame)
            class_name, confidence = predict_currency(processed_frame)

            st.write(f"Prediction: {class_name}, Confidence: {confidence * 100:.2f}%")

            # Break if user stops the app
            if st.button("Stop Webcam"):
                break

        camera.release()

elif option == "IP CCTV":
    st.subheader("Live IP CCTV Prediction")
    ip_url = st.text_input("Enter IP camera URL:", "http://your_ip_camera_url/stream")
    start_button = st.button("Start IP CCTV")

    if start_button and ip_url:
        camera = cv2.VideoCapture(ip_url)
        st_frame = st.empty()

        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access the IP camera.")
                break

            # Show the CCTV feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")

            # Preprocess and predict
            processed_frame = preprocess_image(frame)
            class_name, confidence = predict_currency(processed_frame)

            st.write(f"Prediction: {class_name}, Confidence: {confidence * 100:.2f}%")

            # Break if user stops the app
            if st.button("Stop IP CCTV"):
                break

        camera.release()

elif option == "Upload Image":
    st.subheader("Image Upload Prediction")
    uploaded_file = st.file_uploader("Upload an image of currency:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL Image to OpenCV format
        image = np.array(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        class_name, confidence = predict_currency(processed_image)

        st.write(f"Prediction: {class_name}, Confidence: {confidence * 100:.2f}%")
