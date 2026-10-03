
import os
import time
from datetime import datetime

import cv2
from PIL import Image
from transformers import pipeline

# Force locally downloaded models only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_DIR = "models/yolos-tiny"
ALERT_DIR = "alerts"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
CONFIDENCE_THRESHOLD = 0.70
DETECT_EVERY_N_FRAMES = 3
SNAPSHOT_COOLDOWN_SECONDS = 30

os.makedirs(ALERT_DIR, exist_ok=True)


def main():
    print("Loading local person-detection model...")

    detector = pipeline(
        "object-detection",
        model=MODEL_DIR,
        device=-1,
    )

    print("Model loaded successfully.")
    print("Opening webcam...")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    last_detections = []
    frame_count = 0
    last_snapshot_time = 0
    previous_time = time.time()

    print("Live AI CCTV started.")
    print("Press Q to quit.")
    print("Press S to save a snapshot manually.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            original_height, original_width = frame.shape[:2]
            new_height = int(
                original_height * FRAME_WIDTH / original_width
            )

            frame = cv2.resize(
                frame, (FRAME_WIDTH, new_height)
            )

            # Run AI detection only on selected frames.
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)

                detections = detector(
                    image,
                    threshold=CONFIDENCE_THRESHOLD,
                )

                last_detections = []

                for item in detections:
                    if item["label"].lower() != "person":
                        continue

                    box = item["box"]

                    x1 = max(0, int(box["xmin"]))
                    y1 = max(0, int(box["ymin"]))
                    x2 = min(frame.shape[1], int(box["xmax"]))
                    y2 = min(frame.shape[0], int(box["ymax"]))

                    last_detections.append(
                        (x1, y1, x2, y2, float(item["score"]))
                    )

                # Save one snapshot when a person is detected,
                # then wait before saving another automatically.
                now = time.time()

                if (
                    last_detections
                    and now - last_snapshot_time
                    >= SNAPSHOT_COOLDOWN_SECONDS
                ):
                    filename = datetime.now().strftime(
                        "person_%Y%m%d_%H%M%S.jpg"
                    )
                    path = os.path.join(ALERT_DIR, filename)

                    if cv2.imwrite(path, frame):
                        print("Person detected. Snapshot:", path)
                        last_snapshot_time = now

            # Draw the latest person detections.
            for x1, y1, x2, y2, confidence in last_detections:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                label = f"Person: {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | People: {len(last_detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Offline AI CCTV", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                filename = datetime.now().strftime(
                    "manual_%Y%m%d_%H%M%S.jpg"
                )
                path = os.path.join(ALERT_DIR, filename)

                if cv2.imwrite(path, frame):
                    print("Manual snapshot saved:", path)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("CCTV stopped.")


if __name__ == "__main__":
    main()
