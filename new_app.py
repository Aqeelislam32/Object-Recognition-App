import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image

# Load the model and labels
model = load_model('keras_model.h5')  # Replace with the path to your model
labels = open('labels.txt', 'r').readlines()

# Streamlit app configuration
st.set_page_config(page_title="🔍 Object Recognition", layout="centered")
st.title("📸 Object Recognition and Payment Prediction")

# User selection: CCTV Feed or Image Upload
mode = st.radio("Select Mode", ("CCTV Feed", "Upload Image"))

if mode == "CCTV Feed":
    # Create a placeholder for the CCTV feed
    frame_window = st.image([])

    # Add a start/stop CCTV feed button
    start_camera = st.checkbox("Start CCTV Feed")

    if start_camera:
        # Assuming the CCTV feed is an RTSP stream or a local video file
        cctv_url = st.text_input("Enter CCTV Stream URL", "rtsp://your_cctv_stream_url")  # Replace with actual CCTV stream URL
        camera = cv2.VideoCapture(cctv_url)

        if not camera.isOpened():
            st.error("Unable to open CCTV feed. Please check the stream URL or connection.")
        else:
            stop_camera = False  # Control variable for stopping the loop

            while not stop_camera:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture image from CCTV feed.")
                    break

                # Resize and preprocess the frame for prediction
                resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                processed_frame = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
                processed_frame = (processed_frame / 127.5) - 1

                # Predict with the model
                probabilities = model.predict(processed_frame)
                predicted_label = labels[np.argmax(probabilities)].strip()

                # Assuming the model also returns a prediction for amount paid and change
                # Modify the model prediction logic based on your model's output format
                amount_paid = np.round(probabilities[0][0], 2)  # Modify as per your model
                change_returned = np.round(probabilities[0][1], 2)  # Modify as per your model

                # Display the live CCTV feed with prediction
                annotated_frame = frame.copy()
                cv2.putText(
                    annotated_frame,
                    f"Prediction: {predicted_label} | Amount Paid: {amount_paid} | Change: {change_returned}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                frame_window.image(annotated_frame, channels="BGR", use_column_width=True)

                # Stop CCTV session
                stop_camera_button = st.button("Stop CCTV Feed", key="stop_cctv")
                if stop_camera_button:
                    stop_camera = True

            # Release resources
            camera.release()
            cv2.destroyAllWindows()

elif mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the uploaded image
        if st.button("Submit"):
            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Resize and preprocess the image
            resized_image = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_AREA)
            processed_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
            processed_image = (processed_image / 127.5) - 1

            # Predict with the model
            probabilities = model.predict(processed_image)
            predicted_label = labels[np.argmax(probabilities)].strip()

            # Assuming the model also predicts the amount paid and change
            amount_paid = np.round(probabilities[0][0], 2)  # Modify as per your model
            change_returned = np.round(probabilities[0][1], 2)  # Modify as per your model

            # Display prediction results
            st.write(f"### Prediction: {predicted_label}")
            st.write(f"### Amount Paid: {amount_paid}")
            st.write(f"### Change Returned: {change_returned}")
            st.write(f"### Probabilities: {list(probabilities[0])}")
  
