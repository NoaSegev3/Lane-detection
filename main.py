import cv2
from enum import Enum
from lane_detection.daytime_lane_detector import process_frame_daytime
from lane_detection.nighttime_lane_detector import process_frame_nighttime
from curve_detection.curve_lane_detector import process_frame_curve
from vehicle_detection.vehicle_detector import process_frame_vehicle_detection

class VideoType(Enum):
    DAY_TIME = 1
    NIGHT_TIME = 2
    DETECT_CURVES = 3
    VEHICLE_DETECTION = 4

    def get_video_name(self):
        if self == VideoType.DAY_TIME:
            return r'videos\day_time_video.mp4'
        elif self == VideoType.NIGHT_TIME:
            return r'videos\night_time_video.mp4'
        elif self == VideoType.DETECT_CURVES:
            return r'videos\curve_video.mp4'
        elif self == VideoType.VEHICLE_DETECTION:  
            return r'videos\day_time_video.mp4'

class LaneDetector:
    def __init__(self, video_type):
        self.video_type = video_type

    def process_frame(self, frame):
        if self.video_type == VideoType.DAY_TIME:
            return process_frame_daytime(frame)
        elif self.video_type == VideoType.NIGHT_TIME:
            return process_frame_nighttime(frame)
        elif self.video_type == VideoType.DETECT_CURVES:
            return process_frame_curve(frame)
        elif self.video_type == VideoType.VEHICLE_DETECTION:
            return process_frame_vehicle_detection(frame) 
def main(mode):
    video_path = mode.get_video_name()
    cap = cv2.VideoCapture(video_path)
    detector = LaneDetector(mode)

    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}.")
        return

    print(f"Processing video: {video_path} in mode: {mode.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame = cv2.resize(frame, (960, 540))

        processed_frame = detector.process_frame(frame)
        cv2.imshow(f'{mode.name} Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    mode_input = sys.argv[1] if len(sys.argv) > 1 else "LANE_CHANGE"
    try:
        mode = VideoType[mode_input]
        main(mode)
    except KeyError:
        print(f"Invalid mode '{mode_input}'. Available modes are:")
        for vt in VideoType:
            print(f"  - {vt.name}")