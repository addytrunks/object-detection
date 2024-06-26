from ultralytics import YOLO
import cv2
import cvzone
from sort import *

cap = cv2.VideoCapture('../images/cars.mp4')

model = YOLO('../yolo-weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("../images/mask.png")

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [350, 310, 673, 310]

totalCount = []

while True:
    succ, img = cap.read()

    # if i dont do this, size and type mismatch error occurs between mask and img
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    if img.dtype != mask.dtype:
        mask = mask.astype(img.dtype)

    # overlay mask on our main image
    imgRegion = cv2.bitwise_and(img, mask)

    """Using stream=True enables: 
        Memory Efficiency 
        Real-Time Processing (By processing frames one-by-one,  
        the model can provide immediate results without waiting for the entire video or image set to be processed.)
        Scalability            
    """
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    # Looping through each detected objects
    for r in results:
        boxes = r.boxes
        # Looping through bounding boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class name
            cls = box.cls[0]
            currentClass = classNames[int(cls)]

            if (
                    currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike") and (
                    conf > 0.3):
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0), l=9, rt=2)
        cx, cy = (x1 + w) // 2, (y1 + h) // 2

        # If the centre of the car touches the line, it's a count
        if limits[0] < cx+100 < limits[2] and limits[1] - 160 < cy < limits[1] + 160:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
