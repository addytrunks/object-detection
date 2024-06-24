from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)

# width
cap.set(3, 1280)

# height
cap.set(4, 720)

model = YOLO('./yolo-weights/yolov8l.pt')

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

while True:
    succ, img = cap.read()

    """Using stream=True enables: 
        Memory Efficiency 
        Real-Time Processing (By processing frames one-by-one,  
        the model can provide immediate results without waiting for the entire video or image set to be processed.)
        Scalability            
    """
    results = model(img, stream=True)

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
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(220, 20, 60), colorC=(255, 0, 0))

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class name
            cls = box.cls[0]

            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1-10)),
                               font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, thickness=2, colorR=(255, 255, 0))
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
