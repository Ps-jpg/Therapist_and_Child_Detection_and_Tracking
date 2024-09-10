import os
import cv2
import numpy as np
import argparse
from roboflow import Roboflow
import sys

# Add the sort-master directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sort_dir = os.path.join(current_dir, 'sort-master')
sys.path.append(sort_dir)

# Import Sort
from sort import Sort

# Initialize Roboflow with your API key
rf = Roboflow(api_key="PUXSE0Wcz4NrQf4xpfsK")
project = rf.workspace().project("first-po0yz")
model = project.version(3).model  # Replace '3' with your version number if different

class PersonTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.track_history = {}

    def update(self, detections):
        if len(detections) == 0:
            return []

        detection_array = np.array([[d[2], d[3], d[4], d[5], d[7]] for d in detections])
        tracked_objects = self.tracker.update(detection_array)

        results = []
        for track in tracked_objects:
            track_id = int(track[4])
            bbox = track[:4]
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'class': next((d[6] for d in detections if self.iou(bbox, d[2:6]) > 0.5), 'unknown'),
                    'last_seen': 0
                }
            else:
                self.track_history[track_id]['last_seen'] = 0

            results.append((centroid[0], centroid[1], bbox[0], bbox[1], bbox[2], bbox[3],
                            self.track_history[track_id]['class'], 1.0, track_id))

        # Handle disappeared tracks
        for track_id in list(self.track_history.keys()):
            if self.track_history[track_id]['last_seen'] > 60:  # If not seen for 2 seconds (assuming 30 fps)
                del self.track_history[track_id]
            else:
                self.track_history[track_id]['last_seen'] += 1

        return results

    @staticmethod
    def iou(bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

def process_video(video_path):
    video_path_out = f'{os.path.splitext(video_path)[0]}_out.mp4'

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get frame dimensions and create a video writer for output
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    threshold = 0.5  # Confidence threshold
    person_tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict using Roboflow on the current frame
        response = model.predict(frame, confidence=40, overlap=30).json()

        # Detections
        detections = []
        for pred in response['predictions']:
            x1 = int(pred['x'] - pred['width'] / 2)
            y1 = int(pred['y'] - pred['height'] / 2)
            x2 = int(pred['x'] + pred['width'] / 2)
            y2 = int(pred['y'] + pred['height'] / 2)
            conf = pred['confidence']
            class_name = pred['class']

            if conf > threshold:
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                detections.append((centroid_x, centroid_y, x1, y1, x2, y2, class_name, conf))

        # Update trackers
        tracked_objects = person_tracker.update(detections)

        # Draw bounding boxes and track IDs on the frame
        for obj in tracked_objects:
            centroid_x, centroid_y, x1, y1, x2, y2, class_name, conf, track_id = obj
            color = (0, 255, 0) if class_name == 'adult' else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID: {track_id} | {class_name.capitalize()} | Conf: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (255, 0, 0), -1)

        # Write processed frame to output
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved at: {video_path_out}")

def main():
    parser = argparse.ArgumentParser(description="Video tracking using Roboflow and OpenCV")
    parser.add_argument('--file_name', type=str, help='Specify the video file to scan and track')
    args = parser.parse_args()

    VIDEOS_DIR = './videos'

    if args.file_name:
        video_path = os.path.join(VIDEOS_DIR, args.file_name)
        if os.path.exists(video_path):
            print(f"Processing video: {args.file_name}")
            process_video(video_path)
        else:
            print(f"File {args.file_name} does not exist.")
    else:
        video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            print("No video files found in the videos folder.")
            return
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            process_video(os.path.join(VIDEOS_DIR, video_file))

if __name__ == '__main__':
    main()