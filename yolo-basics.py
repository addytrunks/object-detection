"""
the easiest way to use YOLOv8 for object detections
"""

from ultralytics import YOLO
import cv2

# will use the existing weights (or) it'll be downloaded
model = YOLO('./yolo-weights/yolov8l.pt')
results = model('images/bus.jpg', show=True)
cv2.waitKey(0)
