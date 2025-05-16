import cv2
import os
import random
from glob import glob

# Configuration
output_dir = 'dataset/images'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

val_ratio = 0.2  # 20% validation, 80% training

# Find the highest existing image number
def get_last_image_number(directory):
    images = glob(os.path.join(directory, 'image_*.jpg'))
    if not images:
        return 0
    last_num = max([int(os.path.basename(img).split('_')[1].split('.')[0]) for img in images])
    return last_num + 1

# Start counter from the next available number
img_counter = max(
    get_last_image_number(train_dir),
    get_last_image_number(val_dir)
)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press SPACE to capture an image. Press ESC to exit.")

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
        if random.random() < val_ratio:
            save_dir = val_dir
            set_name = 'validation'
        else:
            save_dir = train_dir
            set_name = 'training'
        
        img_name = f"image_{img_counter:04}.jpg"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, resized_frame)
        print(f"{img_path} saved to {set_name} set!")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()