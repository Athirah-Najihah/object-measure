import cv2
import numpy as np
import random  # Import the random module
import os

# Constants for object detection
thres = 0.5 
nms_threshold = 0.2 

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Charuco board settings
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03  # meters
MARKER_LENGTH = 0.015  # meters

# Load calibration data
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Load Charuco board
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params = cv2.aruco.DetectorParameters()

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image."""
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image, resize)
        image_overlay = cv2.resize(image_overlay, resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect_obstacles(img):
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    detected_objects = []

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        box_width = w
        box_height = h
        detected_object = classNames[classIds[i]-1].upper()
        annotation_text = f"{detected_object} ({box_width}px x {box_height}px)"

        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, annotation_text, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        detected_objects.append((detected_object, x, y, x + w, y + h))

    return detected_objects, img

def get_real_world_distance(p1, p2, rvec, tvec):
    """Convert image points to real-world lengths using the pose of the Charuco board."""
    # Convert image points to 3D points
    p1_3d = np.array([[p1[0], p1[1], 0]], dtype=np.float32)
    p2_3d = np.array([[p2[0], p2[1], 0]], dtype=np.float32)

    # Project 3D points to the image plane
    points_3d, _ = cv2.projectPoints(np.array([p1_3d[0], p2_3d[0]]), rvec, tvec, camera_matrix, np.zeros(5))
    
    # Compute the distance in pixels
    distance_pixels = np.linalg.norm(points_3d[0][0] - points_3d[1][0])
    
    # Convert pixel distance to real-world units (adjust based on your scale)
    conversion_factor = 0.03  # Example factor, adjust based on your setup
    distance_real_world = distance_pixels * conversion_factor

    return distance_real_world

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use your camera index

while True:
    success, frame = cap.read()
    if not success:
        break

    detected_objects, frame = detect_obstacles(frame)

    # Detect Charuco board and estimate pose
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
    if marker_ids is not None and len(marker_ids) > 0:
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, board)
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
            if retval:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=2)

                for obj_name, x1, y1, x2, y2 in detected_objects:
                    top_left = (x1, y1)
                    bottom_right = (x2, y2)

                    # Calculate the width and height in real-world coordinates
                    width_real_world = get_real_world_distance(top_left, (bottom_right[0], top_left[1]), rvec, tvec)
                    height_real_world = get_real_world_distance(top_left, (top_left[0], bottom_right[1]), rvec, tvec)

                    # Display the real-world dimensions
                    cv2.putText(frame, f'Width: {width_real_world:.2f} m', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f'Height: {height_real_world:.2f} m', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('Object Detection with Measurements', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
