import cv2
import numpy as np

left_lane_avg = None
right_lane_avg = None

# Detects edges in the image for nighttime conditions
def canny_edge_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 150)
    return edges

# Isolates the region of interest in the image
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),  
        (int(0.75 * width), int(0.5 * height)),  
        (int(0.25 * width), int(0.5 * height)) 
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

# Detects line segments using the Hough transform
def detect_lines(edges):
    return cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi/180,
        threshold=20,  
        minLineLength=40,  
        maxLineGap=200 
    )

# Calculates the average slope and intercept for detected lines
def average_slope_intercept(lines, width, height):
    global left_lane_avg, right_lane_avg
    left_fit = []
    right_fit = []
    vanishing_y = int(height * 0.5)
    vanishing_x = width // 2
    vanishing_margin = width * 0.1  
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            horizon_x = int((vanishing_y - intercept) / slope) if slope != 0 else x1
            if abs(horizon_x - vanishing_x) <= vanishing_margin:
                if slope < -0.3:  
                    left_fit.append((slope, intercept, length))
                elif slope > 0.3: 
                    right_fit.append((slope, intercept, length))
    if left_fit:
        total_length = sum(length for _, _, length in left_fit)
        left_slope = sum(slope * length for slope, _, length in left_fit) / total_length
        left_intercept = sum(intercept * length for _, intercept, length in left_fit) / total_length
        left_lane = (left_slope, left_intercept)
    else:
        left_lane = None
        
    if right_fit:
        total_length = sum(length for _, _, length in right_fit)
        right_slope = sum(slope * length for slope, _, length in right_fit) / total_length
        right_intercept = sum(intercept * length for _, intercept, length in right_fit) / total_length
        right_lane = (right_slope, right_intercept)
    else:
        right_lane = None
    if left_lane is not None and right_lane is not None:
        left_slope, left_intercept = left_lane
        right_slope, right_intercept = right_lane
        intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
        intersection_y = left_slope * intersection_x + left_intercept
        if abs(intersection_x - vanishing_x) > vanishing_margin or intersection_y < vanishing_y * 0.4:
            left_slope = (vanishing_y - height) / (vanishing_x - 0)
            right_slope = (vanishing_y - height) / (width - vanishing_x)
            left_intercept = height - left_slope * 0
            right_intercept = height - right_slope * width
            left_lane = (left_slope, left_intercept)
            right_lane = (right_slope, right_intercept)
    left_lane_avg = smooth_line(left_lane_avg, left_lane, alpha=0.85)
    right_lane_avg = smooth_line(right_lane_avg, right_lane, alpha=0.85)
    return left_lane_avg, right_lane_avg

# Smooths the detected lines for stability
def smooth_line(previous, current, alpha=0.7):
    if current is None:
        return previous
    if previous is None:
        return current
    return alpha * np.array(previous) + (1 - alpha) * np.array(current)

# Calculates the coordinates of a line segment
def make_coordinates(line, width, height):
    if line is None:
        return None
    slope, intercept = line
    vanishing_y = int(height * 0.5)
    vanishing_x = width // 2
    y1 = height
    y2 = vanishing_y
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    if x2 < vanishing_x:
        x2 = max(x2, vanishing_x - width * 0.1)
    else:
        x2 = min(x2, vanishing_x + width * 0.1)
    return (x1, y1, x2, y2)

# Draws lanes and highlights the area between them
def draw_lines_and_lane(image, lines, width, height):
    line_image = np.zeros_like(image)
    lane_image = np.zeros_like(image)
    
    left_lane, right_lane = average_slope_intercept(lines, width, height)
    left_coords = make_coordinates(left_lane, width, height)
    right_coords = make_coordinates(right_lane, width, height)
    
    if left_coords is not None:
        cv2.line(line_image, (left_coords[0], left_coords[1]), 
                (left_coords[2], left_coords[3]), (0, 255, 0), 8)
    if right_coords is not None:
        cv2.line(line_image, (right_coords[0], right_coords[1]), 
                (right_coords[2], right_coords[3]), (0, 255, 0), 8)
    if left_coords is not None and right_coords is not None:
        base_width = width * 0.1 
        lane_points = np.array([
            [left_coords[0] - int(base_width), left_coords[1]],  
            [left_coords[2] - int(base_width * 0.3), left_coords[3]], 
            [right_coords[2] + int(base_width * 0.3), right_coords[3]], 
            [right_coords[0] + int(base_width), right_coords[1]]  
        ], dtype=np.int32)
        cv2.fillPoly(lane_image, [lane_points], (0, 255, 255))
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    final_image = cv2.addWeighted(combined_image, 1, lane_image, 0.4, 1)
    return final_image

# Processes a single video frame
def process_frame_nighttime(frame):
    height, width, _ = frame.shape
    edges = canny_edge_detection(frame)
    roi = region_of_interest(edges)
    lines = detect_lines(roi)
    lane_image = draw_lines_and_lane(frame, lines, width, height)
    return lane_image