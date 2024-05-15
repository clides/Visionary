from ultralytics import YOLO
import cv2
import cvzone
import math
import time
 
# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("CarProject1.mp4")  # For Video
 
 
model = YOLO("yolov8n.pt")
 
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
 
prev_frame_time = 0
new_frame_time = 0

videoWidth = cap.get(3)
videoHeight = cap.get(4)
# print(videoWidth, videoHeight) # width = 1280 height = 720

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, device="mps") # uses metal (aka mac gpu) if you are using windows and have nvidea gpu, install cuda and config it accordingly

    # Draw lines to represent the boundaries for flagging nearby vehicles
    leftLineFirst = (50, 550)
    leftLineLast = (500, 550)
    rightLineFirst = (780, 550)
    rightLineLast = (1230, 550)
    cv2.line(img, leftLineFirst, leftLineLast, (255, 0, 0), 3)
    cv2.line(img, rightLineFirst, rightLineLast, (255, 0, 0), 3)

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Add if statements here to change color and do other functions if car crosses line
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
        
            # Class Name
            cls = int(box.cls[0])
            className = classNames[cls]

            cvzone.putTextRect(img, f'{className} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(144,238,144))
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (30, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)
