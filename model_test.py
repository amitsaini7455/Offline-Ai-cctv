
import os
import cv2
from PIL import Image
from transformers import pipeline

MODEL_DIR = "models/yolos-tiny"

print("Step 1: Checking model folder...")

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"Model folder not found: {MODEL_DIR}")

print("Step 2: Loading AI person-detection model...")
detector = pipeline(
    "object-detection",
    model=MODEL_DIR,
    device=-1,  # CPU
)

print("Step 3: Opening webcam...")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

print("Step 4: Capturing one frame...")
success, frame = camera.read()
camera.release()

if not success:
    raise SystemExit("Could not capture a frame.")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(rgb)

print("Step 5: Running AI detection...")
results = detector(image, threshold=0.50)

print("\nDetection results:")
if not results:
    print("No objects detected in this frame.")
else:
    for item in results:
        print(
            "Label:", item["label"],
            "| Confidence:", round(item["score"], 3),
            "| Box:", item["box"]
        )

print("\nAI model test completed.")
