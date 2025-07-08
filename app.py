import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- Configuration ---
# Load the YOLOv8 model from the 'best.pt' file
model = YOLO('best.pt')
CONFIDENCE_THRESHOLD = 0.5

def detect_license_plate(input_image):
    """
    This function takes an input image, performs license plate detection,
    draws bounding boxes on it, and returns the resulting image.
    """
    # Perform inference on the input image
    results = model(input_image)

    # Copy the input image to draw on
    img_with_boxes = np.array(input_image)

    # Iterate through the detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            # Draw the bounding box only if confidence is above the threshold
            if conf > CONFIDENCE_THRESHOLD:
                label = result.names[int(box.cls[0])]
                label_text = f'{label}: {conf:.2f}'

                # Draw the rectangle
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the label text above the rectangle
                cv2.putText(img_with_boxes, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return Image.fromarray(img_with_boxes)

# --- Create the Gradio Interface ---
# Define the input and output components for the Gradio app
iface = gr.Interface(
    fn=detect_license_plate,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="🚗 License Plate Detector",
    description="Upload an image of a vehicle, and this app will use a YOLOv8 model to detect the license plate.",
    examples=[["car1.jpg"], ["car2.jpg"]] # You can add example images to your repo
)

# Launch the Gradio app
iface.launch()
