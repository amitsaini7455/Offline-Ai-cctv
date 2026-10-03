
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing camera index from 0 to 1.")
    raise SystemExit(1)

print("Webcam opened successfully!")
print("Press Q to close the camera window.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("Webcam Test - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
