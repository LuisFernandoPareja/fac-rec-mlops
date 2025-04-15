import cv2
import os
from PIL import Image

# Configuration
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)
cam = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press SPACE to capture an image. Press ESC to exit.")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    resized_frame = cv2.resize(frame, (640, 640))
    cv2.imshow("Press SPACE to Capture", resized_frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"{output_dir}/image_{img_counter:04}.jpg"
        cv2.imwrite(img_name, resized_frame)
        print(f"{img_name} saved!")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
