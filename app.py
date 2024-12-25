import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit app title
st.title("YOLO Model Selector and Predictor")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Choose the YOLO model:",
    ("YOLOv8", "YOLOv10")
)

# Load the selected model
if model_choice == "YOLOv8":
    model_path = "C:/Data Science/data science project/rice-leaf-disease-detection/yolov8n.pt"  
elif model_choice == "YOLOv10":
    model_path = "C:/Data Science/data science project/rice-leaf-disease-detection/yolov10n.pt"  

try:
    model = YOLO(model_path)
    st.sidebar.success(f"{model_choice} model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Predict using the YOLO model
    if st.button("Predict"):
        with st.spinner("Running predictions..."):
            results = model.predict(task="detect", source=img_bgr)

            # Display results
            for result in results:
                annotated_img = result.plot()  
                st.image(
                    cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                    caption="Prediction Results",
                    use_column_width=True,
                )

                # Display detection details
                st.subheader("Detection Details")
                st.write("Bounding Boxes:", result.boxes.xyxy.numpy())
                st.write("Confidence Scores:", result.boxes.conf.numpy())
                st.write("Class Indices:", result.boxes.cls.numpy())

else:
    st.info("Please upload an image to start predictions.")