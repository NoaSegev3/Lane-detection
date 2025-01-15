import cv2
import numpy as np
from collections import deque

LEFT_POLY_BUFFER = deque(maxlen=10)
RIGHT_POLY_BUFFER = deque(maxlen=10)

# Applies a mask to keep only the region of interest in the image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

# Defines the vertices of the region of interest for lane detection
def get_lane_region(frame):
    height, width = frame.shape[:2]
    region = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.75 * height)),
        (int(0.6 * width), int(0.75 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)
    return region

# Applies Canny edge detection to an image
def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

# Fits a second-degree polynomial to a set of points
def fit_lane_lines(points):
    if len(points) >= 3:
        y = [p[1] for p in points]
        x = [p[0] for p in points]
        return np.polyfit(y, x, 2)  # Second-degree polynomial
    return None

# Generates lane line points based on a polynomial
def generate_lane_points(poly_coeff, y_start, y_end):
    y_vals = np.linspace(y_start, y_end, num=100, dtype=int)
    x_vals = np.polyval(poly_coeff, y_vals).astype(int)
    return np.column_stack((x_vals, y_vals))

# Smooths the polynomial coefficients using a buffer
def smooth_poly(poly, buffer):
    if poly is not None:
        buffer.append(poly)
    if len(buffer) > 0:
        return np.mean(buffer, axis=0)
    return None

# Ensures the left and right lane polynomials do not overlap
def enforce_lane_consistency(left_poly, right_poly, y_start, y_end):
    if left_poly is None or right_poly is None:
        return left_poly, right_poly

    left_points = generate_lane_points(left_poly, y_start, y_end)
    right_points = generate_lane_points(right_poly, y_start, y_end)

    for lp, rp in zip(left_points, right_points):
        if lp[0] >= rp[0]:
            return None, None
    return left_poly, right_poly

# Calculates the curvature of a polynomial at a given y-value
def calculate_curvature(poly_coeff, y):
    if poly_coeff is None:
        return None
    a, b, _ = poly_coeff
    return ((1 + (2 * a * y + b) ** 2) ** 1.5) / abs(2 * a)

# Draws the lane lines on the frame
def draw_lanes(frame, left_poly, right_poly, height):
    overlay = np.zeros_like(frame)
    y_start = height
    y_end = int(height * 0.75)

    if left_poly is not None:
        left_line = generate_lane_points(left_poly, y_start, y_end)
        cv2.polylines(overlay, [left_line], isClosed=False, color=(0, 0, 255), thickness=10)  # Red

    if right_poly is not None:
        right_line = generate_lane_points(right_poly, y_start, y_end)
        cv2.polylines(overlay, [right_line], isClosed=False, color=(0, 0, 255), thickness=10)  # Red
    return overlay

# Draws the lane area (region between lanes) on the frame
def draw_lane_area(frame, left_poly, right_poly, height):
    overlay = np.zeros_like(frame)
    y_start = height
    y_end = int(height * 0.75)

    if left_poly is not None and right_poly is not None:
        left_points = generate_lane_points(left_poly, y_start, y_end)
        right_points = generate_lane_points(right_poly, y_start, y_end)
        points = np.vstack([left_points, right_points[::-1]])
        cv2.fillPoly(overlay, [points], (255, 255, 255))  # White
    return overlay

# Processes a single video frame
def process_frame_curve(frame):
    height, width = frame.shape[:2]
    lane_region = get_lane_region(frame)
    edges = canny_edge_detection(frame)
    cropped_edges = region_of_interest(edges, lane_region)
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=200
    )

    left_points, right_points = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if 0.5 < abs(slope) < 2:
                if slope < 0:  
                    left_points.extend([(x1, y1), (x2, y2)])
                else:  
                    right_points.extend([(x1, y1), (x2, y2)])

    left_poly = smooth_poly(fit_lane_lines(left_points), LEFT_POLY_BUFFER)
    right_poly = smooth_poly(fit_lane_lines(right_points), RIGHT_POLY_BUFFER)

    left_poly, right_poly = enforce_lane_consistency(left_poly, right_poly, height, int(height * 0.75))

    lane_overlay = draw_lanes(frame, left_poly, right_poly, height)
    lane_area = draw_lane_area(frame, left_poly, right_poly, height)

    combined = cv2.addWeighted(lane_area, 0.3, frame, 0.7, 0)
    final_output = cv2.addWeighted(combined, 0.8, lane_overlay, 1, 1)
    return final_output
