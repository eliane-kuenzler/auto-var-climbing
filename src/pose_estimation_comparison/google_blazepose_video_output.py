import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

"""
This script generates an output video with the skeleton coordinates from MediaPipe.
Enables to check how accurate the MediaPipe pose estimation model is.

Input:
- Climbing video

Output:
- Video with skeleton overlay
"""

# File paths
file_name = "n43_noplus_15"

input_video_path = f'../data/input/videos/{file_name}.mp4'
input_landmarks_path = f'../data/input/landmarks/{file_name}_coordinates_local.parquet'
output_overlay_path = Path(f'../data/output/{file_name}_blazepose.mp4')

# Load landmarks
landmarks = pd.read_parquet(str(input_landmarks_path))

# Open the input video
process_capture = cv2.VideoCapture(str(input_video_path))

# Read the first frame to get video properties
ret, frame = process_capture.read()
fps = process_capture.get(cv2.CAP_PROP_FPS)
height, width, channels = frame.shape
out = cv2.VideoWriter(
    str(output_overlay_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(width), int(height)),
)

# MediaPipe connections for a skeleton
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Head
    (0, 4), (4, 5), (5, 6), (6, 8),  # Head
    (9, 10), (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17),  # Left arm
    (12, 14), (14, 16), (16, 18),  # Right arm
    (23, 24),  # Hips
    (11, 23), (12, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32)  # Right leg
]

max_frame_number = min(int(process_capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(landmarks))

for current_frame_number in tqdm(range(max_frame_number)):
    current_landmarks = landmarks.iloc[current_frame_number]

    landmarks_x = (current_landmarks[current_landmarks.index.str.contains('_x')] * int(width)).values
    landmarks_y = (current_landmarks[current_landmarks.index.str.contains('_y')] * int(height)).values

    # Create a mask to skip NaN values
    valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

    # Draw the skeleton by connecting the landmarks
    for start_idx, end_idx in connections:
        if valid_landmarks[start_idx] and valid_landmarks[end_idx]:
            start_point = (int(landmarks_x[start_idx]), int(landmarks_y[start_idx]))
            end_point = (int(landmarks_x[end_idx]), int(landmarks_y[end_idx]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # Draw a white line
            #cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Draw a green line

    """
    # if you want the points as well -> uncomment
    # Draw circles at the landmark positions
    for x, y, valid in zip(landmarks_x, landmarks_y, valid_landmarks):
        if valid:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # Draw a blue circle
    """

    out.write(frame)
    ret, frame = process_capture.read()

process_capture.release()
out.release()
