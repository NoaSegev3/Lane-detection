import cv2
import numpy as np
from lane_detection.lane_utils import region_of_interest, lane_lines, draw_lane_lines
from lane_detection.lane_change_detector import detect_lane_change, display_lane_change_message

prev_lane_center = None
lane_change_threshold = 50  # Threshold for detecting lane changes

# Detects edges in the frame using the Canny edge detection method
def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mean_intensity = np.mean(blurred)
    low_threshold = int(max(50, 0.66 * mean_intensity))
    high_threshold = int(max(100, 1.33 * mean_intensity))
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges

# Detects line segments in the edge-detected image using the Hough Line Transform
def hough_transform_daytime(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=100
    )
    return lines

# Processes a single frame
def process_frame_daytime(frame):
    edges = canny_edge_detection(frame)
    cropped_edges = region_of_interest(edges)
    lines = hough_transform_daytime(cropped_edges)
    if lines is None or len(lines) < 2:
        return frame
    detected_lanes = lane_lines(frame, lines)
    direction, message_counter = detect_lane_change(detected_lanes)
    message_counter = display_lane_change_message(frame, message_counter, direction)
    lane_frame = draw_lane_lines(frame, detected_lanes)
    return lane_frame
