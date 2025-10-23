import cv2
import numpy as np
import os

# Project base path
base_dir = r"D:\Project Btech\Dynamic Signal"
captured_images_dir = r"D:\Project Btech\Dynamic Signal\input_images"

# Load YOLO model
net = cv2.dnn.readNet(
    os.path.join(base_dir, "yolov3.weights"),
    os.path.join(base_dir, "yolov3.cfg")
)

# Load COCO class names
with open(os.path.join(base_dir, "coco.names"), "r") as f:
    classes = f.read().strip().split("\n")

# Get latest captured image
image_files = [f for f in os.listdir(captured_images_dir)
               if f.startswith("captured_image_") and f.endswith(".jpg")]

if not image_files:
    print("No captured images found.")
    exit()

latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(captured_images_dir, x)))
image_path = os.path.join(captured_images_dir, latest_image)

# Load image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
detections = net.forward(output_layers)

# Define vehicle class IDs (car, motorbike, bus, truck, etc.)
vehicle_classes = [1, 2, 3, 5, 7]  # person=0, car=2, etc.

boxes = []
confidences = []
class_ids = []

# Collect bounding boxes and confidences
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id in vehicle_classes:
            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

vehicle_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw final bounding boxes
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    vehicle_count += 1
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), font, 0.5, (0, 255, 0), 2)

# Save output image
output_path = os.path.join(base_dir, "vehicle_detection_output.jpg")
cv2.imwrite(output_path, image)
print(f"Output image saved to: {output_path}")

# Save vehicle count to a file
with open(os.path.join(base_dir, "vehicle_count.txt"), "w") as f:
    f.write(str(vehicle_count))

print(f"Vehicle count detected: {vehicle_count}")
