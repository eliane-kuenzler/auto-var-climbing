import time
import cv2
from pathlib import Path
from rtmlib import Wholebody, draw_skeleton
import numpy as np

"""
This script calculates the FPS and pose detection percentage for RTMO model.

Input:
- Climbing videos

Output:
- Average FPS
- Average Pose Detection Percentage
"""

# Initialize the RTMO model
device = 'cpu'  # or 'cuda' if GPU is available
backend = 'onnxruntime'
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# RTMO model initialization
wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='performance',  # 'balanced', 'performance', 'lightweight'
                      backend=backend, device=device)

# Video directory path
video_path = Path('../../data/input')

# List to store results for each video
video_results = []

# Process each video in the directory
for video_file in list(video_path.glob("*.avi")) + list(video_path.glob("*.mp4")):
    print(f"Starting processing video: {video_file.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_file))

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS from video: {fps}")

    # Initialize timing for FPS calculation
    start_time = time.time()
    frame_count = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # Detect skeleton keypoints and scores
        keypoints, scores = wholebody(frame)

        # Count as a detected frame if any keypoint has a score above the threshold (e.g., 0.5)
        if keypoints is not None and scores is not None:
            if np.any(scores > 0.5):  # At least one keypoint is detected with sufficient confidence
                detected_frames += 1

    # Calculate pose detection percentage
    if frame_count > 0:
        pose_detection_percentage = (detected_frames / frame_count) * 100  # Percentage of frames with detected poses
        print(f"Pose detection percentage: {pose_detection_percentage:.2f}%")
    else:
        pose_detection_percentage = 0

    # Calculate actual FPS based on processing time
    end_time = time.time()
    total_time = end_time - start_time
    fps_calculated = frame_count / total_time
    print(f"Calculated FPS for the video: {fps_calculated:.2f}")

    # Store the result for this video
    video_results.append({
        "video_name": video_file.name,
        "fps": fps_calculated,  # Use calculated FPS here
        "pose_detection_percentage": pose_detection_percentage
    })

    # Close the video capture
    cap.release()

    print(f"Finished processing video: {video_file.name}")

# Calculate averages for RTMO
total_videos = len(video_results)
average_fps = np.mean([result["fps"] for result in video_results])
average_pose_detection_percentage = np.mean([result["pose_detection_percentage"] for result in video_results])

# Output the results for each video
for result in video_results:
    print(
        f"Video: {result['video_name']} | FPS: {result['fps']:.2f} | Pose Detection: {result['pose_detection_percentage']:.2f}%")

# Average results for RTMO
print(f"\nRTMO Overall Average FPS: {average_fps:.2f}")
print(f"RTMO Overall Average Pose Detection Percentage: {average_pose_detection_percentage:.2f}%")
