import cv2
import time
import os

def capture_image(camera_index, captured_images_dir,interval_seconds):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    print("Camera opened successfully.")
    time.sleep(interval_seconds)
    for i in range(5):  # Capture only 5 images
        timestamp = time.strftime("%Y%m%d%H%M%S")
        image_filename = f"captured_image_{timestamp}.jpg"
        image_path = os.path.join(captured_images_dir, image_filename)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return
        cv2.imwrite(image_path, frame)
        print("{i} Image captured and saved as", image_path)
        time.sleep(interval_seconds)
    cap.release()
    print("Image captured and saved as", image_path)

def main():
    camera_index = 0
    interval_seconds = 2

    # Save images to a specific directory
    captured_images_dir = r"D:\Project Btech\Dynamic Signal\input_images"
    os.makedirs(captured_images_dir, exist_ok=True)

    # for i in range(5):  # Capture only 5 images
    #     timestamp = time.strftime("%Y%m%d%H%M%S")
    #     image_filename = f"captured_image_{timestamp}.jpg"
    #     image_path = os.path.join(captured_images_dir, image_filename)
    capture_image(camera_index, captured_images_dir,interval_seconds)
        # time.sleep(interval_seconds)

    print("\nFinished capturing 5 images.")

if __name__ == "__main__":
    main()
