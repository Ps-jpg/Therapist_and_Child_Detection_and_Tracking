import os
import cv2
import numpy as np
from roboflow import Roboflow
import argparse
from tqdm import tqdm
import tkinter as tk
from PIL import Image, ImageTk

# Initialize Roboflow with your API key
api_key = "9dMxf4G2rQNy0BGmjzt6"
model_id = "first-po0yz/3"

# Initialize Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("first-po0yz")
model = project.version(3).model  # Ensure you replace VERSION with the correct version number

# Video paths
VIDEOS_DIR = os.path.join('.', 'videos')

def process_video(video_path):
    video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0])

    # Open video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Failed to read the video file.")
        return

    # Get frame dimensions
    H, W, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    threshold = 0.8  # Detection confidence threshold
    frame_limit = float('inf')  # Process the entire video
    frame_counter = 0

    # Initialize tracking variables
    trackers = []
    track_ids = []
    kalman_filters = []
    next_id = 1
    last_seen = {}

    # Create a Tkinter window
    window = tk.Tk()
    window.title("Video Tracking")

    # Create a label to display the video
    label = tk.Label(window)
    label.pack()

    # Loop through video frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")

    while ret:
        try:
            # Predict using Roboflow on the current frame
            response = model.predict(frame, confidence=40, overlap=30).json()
        except Exception as e:
            print(f"Error predicting frame: {e}")
            ret, frame = cap.read()
            continue

        # Create a list to hold the detected objects
        detections = []

        # Process predictions
        for prediction in response['predictions']:
            x1 = int(prediction['x'] - prediction['width'] / 2)
            y1 = int(prediction['y'] - prediction['height'] / 2)
            x2 = int(prediction['x'] + prediction['width'] / 2)
            y2 = int(prediction['y'] + prediction['height'] / 2)
            conf = prediction['confidence']
            class_name = prediction['class']

            if conf > threshold:
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                detections.append((centroid_x, centroid_y, x1, y1, x2, y2, class_name, conf))

        # Update existing trackers and assign new ones
        new_trackers = []
        for (centroid_x, centroid_y, x1, y1, x2, y2, class_name, conf) in detections:
            matched = False
            for i, (tracker, kalman_filter) in enumerate(zip(trackers, kalman_filters)):
                if np.linalg.norm(
                        np.array(tracker[:2]) - np.array([centroid_x, centroid_y])) < 30:  # Adjust threshold as needed
                    trackers[i] = (centroid_x, centroid_y, x1, y1, x2, y2)
                    kalman_filter.predict()
                    measurement = np.array([centroid_x, centroid_y, 0, 0], dtype=np.float32)
                    kalman_filter.correct(measurement)
                    matched = True
                    break

            if not matched:
                trackers.append((centroid_x, centroid_y, x1, y1, x2, y2))
                track_ids.append(next_id)
                next_id += 1
                kf = cv2.KalmanFilter(4, 4)
                kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float32)
                kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
                kf.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
                kf.measurementNoiseCov = 1e-3 * np.eye(4, dtype=np.float32)
                kf.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
                kf.statePost = np.array([centroid_x, centroid_y, 0, 0], dtype=np.float32)
                kalman_filters.append(kf)

        # Remove trackers that have not been seen for a certain number of frames
        trackers_to_remove = []
        for i, (tracker, track_id) in enumerate(zip(trackers, track_ids)):
            if track_id in last_seen:
                if frame_counter - last_seen[track_id] > 30:  # Adjust the threshold as needed
                    trackers_to_remove.append(i)
            else:
                last_seen[track_id] = frame_counter

        for index in sorted(trackers_to_remove, reverse=True):
            del trackers[index]
            del track_ids[index]
            del kalman_filters[index]

        # Draw bounding boxes and IDs
        for i, (tracker, track_id) in enumerate(zip(trackers, track_ids)):
            centroid_x, centroid_y, x1, y1, x2, y2 = tracker
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame in the Tkinter window
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.config(image=imgtk)
        label.image = imgtk
        window.update()

        # Wait for key press (exit by pressing 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read the next frame
        ret, frame = cap.read()
        pbar.update(1)

    # Release video resources
    cap.release()
    out.release()
    window.destroy()
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Video tracking using Roboflow and OpenCV")
    parser.add_argument('--filename', type=str, help='Specify the video file to scan and track')
    args = parser.parse_args()

    if args.filename:
        video_path = os.path.join(VIDEOS_DIR, args.filename)
        if os.path.exists(video_path):
            print(f"Processing video: {args.filename}")
            process_video(video_path)
        else:
            print(f"File {args.filename} does not exist.")
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
