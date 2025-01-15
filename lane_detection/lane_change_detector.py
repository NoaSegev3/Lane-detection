import cv2
import numpy as np

# Global variables for lane tracking
prev_lane_width = None
prev_lane_center = None
lane_change_message_counter = 0
required_lane_width = None
required_lane_center = None
n_first_lanes = 10
first_lanes_width = []
first_lanes_center = []
frame_counter = 0

# Computes the width of the lane based on detected lane lines
def compute_lane_width(lines):
    if len(lines) != 2:
        raise ValueError(f"Invalid lines format: {lines}")
    if lines[0] is None or lines[1] is None:
        return None  # Handle missing lines gracefully
    # Unpack nested tuples
    (x1, y1), (x2, y2) = lines[0]
    (x3, y3), (x4, y4) = lines[1]
    # Compute lane width
    lane_width = np.abs((x1 + x2) / 2 - (x3 + x4) / 2)
    return lane_width

# Computes the center of the lane based on detected lane lines
def compute_center(lines):
    if len(lines) != 2:
        raise ValueError(f"Invalid lines format: {lines}")
    # Unpack nested tuples
    (x1, y1), (x2, y2) = lines[0]
    (x3, y3), (x4, y4) = lines[1]
    # Compute the center as the midpoint of the left and right lane line bases
    center = ((x1 + x2) / 2 + (x3 + x4) / 2) / 2
    return center

# Removes outliers from a dataset using the IQR method
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

# Detects lane changes based on lane width and center changes
def detect_lane_change(lines):
    global prev_lane_width, prev_lane_center, required_lane_width, required_lane_center
    global frame_counter, n_first_lanes, first_lanes_width, first_lanes_center
    global lane_change_message_counter
    if len(lines) != 2:
        return None, lane_change_message_counter
    # Get the width from the lane lines themselves
    frame_width = max(lines[1][0][0], lines[1][1][0]) - min(lines[0][0][0], lines[0][1][0])
    # Compute the current lane width and center
    current_lane_width = compute_lane_width(lines)
    current_lane_center = compute_center(lines)
    # Collect initial frames for baseline calculation
    if frame_counter < n_first_lanes:
        if current_lane_width is not None:  # Only add valid measurements
            first_lanes_width.append(current_lane_width)
            first_lanes_center.append(current_lane_center)
        frame_counter += 1
        return None, lane_change_message_counter
    if frame_counter == n_first_lanes:
        # Remove outliers before computing baseline
        first_lanes_width = remove_outliers(first_lanes_width)
        first_lanes_center = remove_outliers(first_lanes_center)
        # Compute baseline width and center
        required_lane_width = np.mean(first_lanes_width)
        required_lane_center = np.mean(first_lanes_center)
        frame_counter += 1
    # Add hysteresis to prevent false detections
    if prev_lane_width is not None:
        width_change_rate = abs(current_lane_width - prev_lane_width) / prev_lane_width
        center_change_rate = abs(current_lane_center - prev_lane_center) / frame_width
        # Only trigger if both sudden width change AND center shift
        if (current_lane_width < required_lane_width * 0.8 and  # Strict width threshold
            width_change_rate > 0.15 and  # Significant width change
            center_change_rate > 0.1):    # Significant center shift
            lane_change_message_counter = 100
            if current_lane_center < required_lane_center:
                return "LEFT", lane_change_message_counter
            else:
                return "RIGHT", lane_change_message_counter
    # Update previous values with smoothing
    if prev_lane_width is None:
        prev_lane_width = current_lane_width
        prev_lane_center = current_lane_center
    else:
        # Smooth the measurements
        alpha = 0.8  # Smoothing factor
        prev_lane_width = alpha * prev_lane_width + (1 - alpha) * current_lane_width
        prev_lane_center = alpha * prev_lane_center + (1 - alpha) * current_lane_center
    return None, lane_change_message_counter

# Displays a message on the frame indicating the detected lane change direction
def display_lane_change_message(frame, message_counter, direction):
    if message_counter > 0 and direction is not None:
        if direction == "RIGHT":
            message = "Lane Change Detected!"
        elif direction == "LEFT":
            message = "Lane Change Detected!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        # Draw the text
        cv2.putText(frame, message, (text_x, text_y), font, 1, (0, 255, 0), 2)
        # Draw arrow
        draw_arrow_for_lane_change(frame, direction, (text_x + 430, text_y + 50))
        # Decrease message counter
        message_counter -= 1
    return message_counter

# Draws an arrow on the frame indicating the lane change direction
def draw_arrow_for_lane_change(frame, direction, base_position, arrow_length=100, arrow_color=(0, 255, 0), thickness=5):
    start_point = base_position
    if direction == "LEFT":
        end_point = (base_position[0] - arrow_length, base_position[1])
    else:
        end_point = (base_position[0] + arrow_length, base_position[1])
    cv2.arrowedLine(frame, start_point, end_point, arrow_color, thickness, tipLength=0.3)
