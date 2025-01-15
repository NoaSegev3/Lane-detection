import cv2
import numpy as np
from collections import deque
from lane_detection.daytime_lane_detector import process_frame_daytime

# Parameters
MAX_BUFFER_SIZE = 10
VEHICLE_DETECTION_THRESHOLD = 3
MAX_DETECTED_VEHICLES = 2  
PROXIMITY_REGION = (0.35, 0.65, 0.5, 0.85)  
frame_counter = 0

# A buffer to store detected vehicles for stabilization
vehicle_detection_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# Load Haar Cascade for Vehicle Detection
vehicle_cascade = cv2.CascadeClassifier('vehicle_detection/haarcascade_car.xml')
if vehicle_cascade.empty():
    raise IOError("Failed to load Haar cascade. Check the path: 'vehicle_detection/haarcascade_car.xml'")

# Function to infer the lane mask from the processed lane frame
def infer_lane_mask_from_frame(lane_frame):
    """
    Infer the lane mask from the processed lane frame by thresholding.
    """
    gray = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return lane_mask

# Function to merge overlapping or close bounding boxes
def merge_overlapping_boxes(boxes):
    """
    Merge overlapping or very close bounding boxes into a single box to reduce duplicates.
    """
    merged_boxes = []
    for i, box_a in enumerate(boxes):
        x1_a, y1_a, w_a, h_a = box_a
        for j, box_b in enumerate(boxes):
            if i >= j:
                continue
            x1_b, y1_b, w_b, h_b = box_b

            # Check for overlap
            if (x1_a < x1_b + w_b and x1_a + w_a > x1_b and
                y1_a < y1_b + h_b and y1_a + h_a > y1_b):
                x_min = min(x1_a, x1_b)
                y_min = min(y1_a, y1_b)
                x_max = max(x1_a + w_a, x1_b + w_b)
                y_max = max(y1_a + h_a, y1_b + h_b)
                merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
                break
        else:
            merged_boxes.append(box_a)
    return merged_boxes

# Function to detect vehicles while excluding the lane region
def detect_vehicles(frame, lane_mask):
    """
    Detect vehicles in the frame using Haar cascades and exclude regions occupied by lanes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Exclude lane area using the lane mask
    excluded_frame = cv2.bitwise_and(gray, cv2.bitwise_not(lane_mask))

    # Detect vehicles using Haar cascades
    detections = vehicle_cascade.detectMultiScale(
        excluded_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    height, width = frame.shape[:2]
    x_min, x_max, y_min, y_max = (
        int(PROXIMITY_REGION[0] * width),
        int(PROXIMITY_REGION[1] * width),
        int(PROXIMITY_REGION[2] * height),
        int(PROXIMITY_REGION[3] * height),
    )

    vehicle_boxes = []
    for (x, y, w, h) in detections:
        center_x = x + w // 2
        center_y = y + h // 2

        if x_min < center_x < x_max and y_min < center_y < y_max:
            vehicle_boxes.append((x, y, w, h))

    # Merge overlapping boxes
    vehicle_boxes = merge_overlapping_boxes(vehicle_boxes)

    # Limit to closest detected vehicles
    vehicle_boxes = sorted(vehicle_boxes, key=lambda box: box[1])[:MAX_DETECTED_VEHICLES]

    return vehicle_boxes

# Function to stabilize vehicle detections using a buffer
def stabilize_detections(vehicle_boxes):
    global vehicle_detection_buffer
    vehicle_detection_buffer.append(vehicle_boxes)

    all_boxes = []
    for boxes in vehicle_detection_buffer:
        all_boxes.extend(boxes)

    stable_boxes = []
    for box in all_boxes:
        count = sum([1 for boxes in vehicle_detection_buffer if box in boxes])
        if count >= VEHICLE_DETECTION_THRESHOLD:
            stable_boxes.append(box)

    stable_boxes = list({tuple(box) for box in stable_boxes})[:MAX_DETECTED_VEHICLES]
    return stable_boxes

# Function to draw bounding boxes around detected vehicles
def draw_vehicle_boxes(frame, vehicle_boxes):
    """
    Draw bounding boxes around detected vehicles on the frame.
    """
    for x, y, w, h in vehicle_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Main function to process a single frame for vehicle detection
def process_frame_vehicle_detection(frame):
    """
    Process a single frame to detect vehicles and lanes, and draw results.
    """
    global frame_counter

    lane_frame = process_frame_daytime(frame)
    lane_mask = infer_lane_mask_from_frame(lane_frame)
    vehicle_boxes = detect_vehicles(frame, lane_mask)
    stabilized_boxes = stabilize_detections(vehicle_boxes)
    draw_vehicle_boxes(lane_frame, stabilized_boxes)

    frame_counter += 1
    return lane_frame
