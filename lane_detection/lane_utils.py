import cv2
import numpy as np

prev_left_lane = None
prev_right_lane = None

# Applies a mask to isolate the region of interest in the image
def region_of_interest(edges, vertices=None):
    mask = np.zeros_like(edges)  
    ignore_mask_color = 255
    if vertices is None:
        height, width = edges.shape[:2]
        bottom_left = (int(width * 0.1), int(height * 0.95))
        bottom_right = (int(width * 0.9), int(height * 0.95))
        top_left = (int(width * 0.4), int(height * 0.6))
        top_right = (int(width * 0.6), int(height * 0.6))
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

# Computes the average slope and intercept of left and right lane lines
def average_slope_intercept(lines, width):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    if lines is None:
        return None, None
    min_slope = 0.3  # Filter out near-horizontal lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if abs(slope) < min_slope:
                continue
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = (
        np.dot(left_weights, left_lines) / np.sum(left_weights)
        if len(left_weights) > 0 else None
    )
    right_lane = (
        np.dot(right_weights, right_lines) / np.sum(right_weights)
        if len(right_weights) > 0 else None
    )
    return left_lane, right_lane

# Extrapolates lane line coordinates to extend them across the image
def extrapolate_lines(image, line):
    if line is None:
        return None
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, y1), (x2, y2))

# Stabilizes the detected lane lines using smoothing
def stabilize_lane(new_lane, prev_lane, alpha=0.75):
    if new_lane is None:
        return prev_lane
    if prev_lane is None:
        return new_lane
    return alpha * np.array(prev_lane) + (1 - alpha) * np.array(new_lane)

# Detects and stabilizes lane lines in the image
def lane_lines(image, lines):
    global prev_left_lane, prev_right_lane

    left_lane, right_lane = average_slope_intercept(lines, image.shape[1])
    left_lane = stabilize_lane(left_lane, prev_left_lane)
    right_lane = stabilize_lane(right_lane, prev_right_lane)
    prev_left_lane, prev_right_lane = left_lane, right_lane
    left_line = extrapolate_lines(image, left_lane)
    right_line = extrapolate_lines(image, right_lane)
    return left_line, right_line

# Draws lane lines and the lane area on the image
def draw_lane_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines[0] is not None and lines[1] is not None:
        left_line, right_line = lines
        cv2.line(line_image, *left_line, [0, 0, 255], 10)
        cv2.line(line_image, *right_line, [0, 0, 255], 10)
        points = np.array([
            [left_line[0], left_line[1], right_line[1], right_line[0]]
        ])
        cv2.fillPoly(line_image, [points], [255, 255, 255])
    return cv2.addWeighted(image, 1.0, line_image, 0.6, 0)
