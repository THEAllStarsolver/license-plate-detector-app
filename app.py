import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- Configuration ---
MODEL_PATH = 'best.pt' # This will use the model file in the same folder
CONFIDENCE_THRESHOLD = 0.5

# --- Helper Functions ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def draw_detections(image, results):
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > CONFIDENCE_THRESHOLD:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f'{result.names[int(box.cls[0])]}: {conf:.2f}'
                cv2.putText(img_bgr, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# --- Streamlit App ---
st.set_page_config(page_title="License Plate Detection", layout="wide")
st.title("License Plate Detector 🚗")
st.write("Upload an image, and the model will detect license plates.")

model = load_model(MODEL_PATH)

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            results = model(image)
            image_with_boxes = draw_detections(image, results)
            st.image(image_with_boxes, caption="Image with Detections", use_column_width=True)
else:
    st.warning("Model could not be loaded. Check that 'best.pt' is in the repository.")