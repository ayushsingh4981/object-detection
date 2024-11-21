import cv2
import numpy as np

# Thresholds
thres = 0.45  # Confidence threshold
nms_threshold = 0.2  # Non-max suppression threshold

# Initialize camera
cap = cv2.VideoCapture(0)

# Load class names
classNames = []
classFile = r'C:\Users\ASUS\OneDrive\Desktop\Object_Detection_Files\coco.names'  # Path to coco.names
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Filtered class names (only fruits and vegetables)
fruit_veg_classes = [
    "apple", "banana", "orange", "carrot", "broccoli", "tomato"
]

# Paths to model files
configPath = r'C:\Users\ASUS\OneDrive\Desktop\Object_Detection_Files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to config
weightsPath = r'C:\Users\ASUS\OneDrive\Desktop\Object_Detection_Files\frozen_inference_graph.pb'  # Path to weights

# Load the model
net = cv2.dnn.DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Detection loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Perform detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Convert detection results to lists
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:  # Check if indices are not empty
        for i in indices.flatten():  # Use flatten() for compatibility
            classId = int(classIds[i]) - 1  # Adjust for zero-based indexing
            className = classNames[classId]

            if className in fruit_veg_classes:  # Filter for fruits and vegetables
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(
                    img,
                    className.upper(),
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    # Display the result
    cv2.imshow("Output", img)

    # Break on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
