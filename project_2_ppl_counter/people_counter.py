from ultralytics import YOLO
import cv2
import cvzone
from sort import *

# Open the video file
cap = cv2.VideoCapture('../images/people.mp4')

# Load the YOLO model
model = YOLO('../yolo-weights/yolov8n.pt')

# List of class names that YOLO can detect
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

# Load the mask image
mask = cv2.imread("../images/mask2.png")

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line coordinates for vehicle counting
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

# List to keep track of counted vehicle IDs
totalCountUp = []
totalCountDown = []

while True:
    # Read a frame from the video
    succ, img = cap.read()
    if not succ:
        break

    # Resize and type match mask to the frame if needed
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    if img.dtype != mask.dtype:
        mask = mask.astype(img.dtype)

    # Apply the mask to the frame
    imgRegion = cv2.bitwise_and(img, mask)

    # Perform object detection
    results = model(imgRegion, stream=True)

    # Initialize an empty array for detections
    detections = np.empty((0, 5))

    # Loop through each detected object
    for r in results:
        boxes = r.boxes
        # Loop through each bounding box
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Get confidence score
            conf = round(float(box.conf[0]), 2)

            # Get class name
            cls = box.cls[0]
            currentClass = classNames[int(cls)]

            # Filter detections by class (car, truck, bus, motorbike) and confidence
            if currentClass in ["person"] and conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with the new detections
    resultsTracker = tracker.update(detections)

    # Draw the counting line
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 2)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 2)

    # Loop through the tracker results
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and ID on the frame
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0), l=9, rt=2)
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1 - 10)),
                           font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.5, thickness=3, colorR=(255, 255, 0),
                           offset=10)

        # Calculate the center of the bounding box
        cx, cy = (x1 + w) // 2, (y1 + h) // 2

        # Check if the center of the vehicle crosses the counting line
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                # Change the line color to green when a vehicle is counted
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 2)

        cvzone.putTextRect(img, f' Up: {len(totalCountUp)}', (929, 345), scale=1)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                # Change the line color to green when a vehicle is counted
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 2)

        cvzone.putTextRect(img, f' Down: {len(totalCountDown)}', (1100, 345), scale=1)

    # Display the total count on the frame

    # Show the frame
    cv2.imshow('img', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
