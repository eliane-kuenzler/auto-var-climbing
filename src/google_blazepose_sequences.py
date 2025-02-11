import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import math

"""
This script generates an output video with left wrist, right wrist, and hip center tracks.
It also includes annotations (holds), the frame number and a color legend for the tracks.
The skeleton, hold annotations, and tracks are drawn on a white background.

Input:
- Climbing video
- Landmarks (local coordinates)
- Annotations (holds)

Output:
- Video with tracks, framenumber and annotations
"""

# File paths
file_name = "edited_villars_men_semifinals_n110_plus_14+"

# Automatically generated paths based on the file name
input_video_path = f'../data/input/videos/{file_name}.mp4'
input_landmarks_path = f'../data/input/landmarks/{file_name}_coordinates_local.parquet'
annotations_path = f'../data/input/topos/{file_name}_annotations.json'
output_video_path = f'../data/output/sequences/{file_name}_sequence.mp4'

# Choose the frame range to process
# Set the starting frame and end frame
start_frame = 700
end_frame = 900

# Load the saved annotations
with open(str(annotations_path), 'r') as f:
    annotations = json.load(f)

# Load landmarks
landmarks = pd.read_parquet(input_landmarks_path)

# Open the input video to get frames
capture = cv2.VideoCapture(str(input_video_path))

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

# Define annotation color (for the holds)
color = (0, 0, 0)  # Black for annotation

# Define keypoint dictionary for specific landmarks
keypoint_dict = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

# MediaPipe skeleton connections (simplified for the skeleton)
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

# Initialize tracks for hip center and wrists
hip_track = []
left_hand_track = []
right_hand_track = []

# Create the output video writer (to save the video)
output_video = cv2.VideoWriter(
    str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Colors for the tracks and legend
red = (0, 0, 255)  # Red for Hip Center
blue = (255, 0, 0)  # Blue for Left Wrist
green = (0, 255, 0)  # Green for Right Wrist

# Process frames 1000 to 1600
for frame_idx in range(start_frame, end_frame):
    ret, frame = capture.read()
    if not ret:
        break

    current_landmarks = landmarks.iloc[frame_idx]

    # Scale landmarks to pixel coordinates
    landmarks_x = (current_landmarks[current_landmarks.index.str.contains('_x')] * width).values
    landmarks_y = (current_landmarks[current_landmarks.index.str.contains('_y')] * height).values
    valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

    # Create a white background frame for the zoomed-in section
    zoom_frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw skeleton connections in black (for the skeleton)
    for start_idx, end_idx in connections:
        if valid_landmarks[start_idx] and valid_landmarks[end_idx]:
            start_point = (int(landmarks_x[start_idx]), int(landmarks_y[start_idx]))
            end_point = (int(landmarks_x[end_idx]), int(landmarks_y[end_idx]))
            cv2.line(zoom_frame, start_point, end_point, (0, 0, 0), 2)  # Black for skeleton

    # Track hip center and wrists
    if valid_landmarks[keypoint_dict["left_hip"]] and valid_landmarks[keypoint_dict["right_hip"]]:
        left_hip = (landmarks_x[keypoint_dict["left_hip"]], landmarks_y[keypoint_dict["left_hip"]])
        right_hip = (landmarks_x[keypoint_dict["right_hip"]], landmarks_y[keypoint_dict["right_hip"]])
        hip_center = (
            int((left_hip[0] + right_hip[0]) / 2),
            int((left_hip[1] + right_hip[1]) / 2)
        )
        hip_track.append(hip_center)

    if valid_landmarks[keypoint_dict["left_wrist"]]:
        left_hand = (
            int(landmarks_x[keypoint_dict["left_wrist"]]),
            int(landmarks_y[keypoint_dict["left_wrist"]])
        )
        left_hand_track.append(left_hand)

    if valid_landmarks[keypoint_dict["right_wrist"]]:
        right_hand = (
            int(landmarks_x[keypoint_dict["right_wrist"]]),
            int(landmarks_y[keypoint_dict["right_wrist"]])
        )
        right_hand_track.append(right_hand)

    # Draw the tracks for the hip and wrists on the white background frame
    for i in range(1, len(hip_track)):
        adjusted_hip = (hip_track[i][0], hip_track[i][1])
        cv2.line(zoom_frame, adjusted_hip, adjusted_hip, red, 2)  # Red for hip center

    for i in range(1, len(left_hand_track)):
        adjusted_left_hand = (left_hand_track[i][0], left_hand_track[i][1])
        cv2.line(zoom_frame, adjusted_left_hand, adjusted_left_hand, blue, 2)  # Blue for left hand

    for i in range(1, len(right_hand_track)):
        adjusted_right_hand = (right_hand_track[i][0], right_hand_track[i][1])
        cv2.line(zoom_frame, adjusted_right_hand, adjusted_right_hand, green, 2)  # Green for right hand

    # Draw annotations (holds)
    for annotation in annotations:
        center = tuple(int(coord) for coord in annotation['center'])
        radius = annotation['radius']
        hold_number = annotation['number']

        # Draw the circle in black
        cv2.circle(zoom_frame, center, radius, color, 2)  # Black for circle
        cv2.circle(zoom_frame, center, 2, color, -1)  # Center point

        # Draw the hold number text
        cv2.putText(zoom_frame, hold_number, (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add color legend box at the top-right corner (moved slightly left and down)
    legend_x = width - 220  # Moved 20px left from the right side
    legend_y = 40  # Moved 20px down from the top
    legend_width = 180
    legend_height = 130  # Slightly increased height to accommodate larger text
    cv2.rectangle(zoom_frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height),
                  (255, 255, 255), -1)

    # Add text to explain the color codes using actual colors for the text
    cv2.putText(zoom_frame, 'Tracking Legend:', (legend_x - 45, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Larger text for the header

    # Red, Blue, Green for tracks (using actual colors)
    cv2.putText(zoom_frame, 'Red: Hip Center', (legend_x - 45, legend_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)  # Larger text
    cv2.putText(zoom_frame, 'Blue: Left Wrist', (legend_x - 45, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)  # Larger text
    cv2.putText(zoom_frame, 'Green: Right Wrist', (legend_x - 45, legend_y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)  # Larger text

    # Display the frame number in the left bottom corner
    frame_number_text = f"Frame: {frame_idx}"
    cv2.putText(zoom_frame, frame_number_text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black text with a border

    # Write the frame to the output video
    output_video.write(zoom_frame)

# Release resources
capture.release()
output_video.release()

print("Sequence of interest with skeleton, tracks, annotations and frame number saved to mp4.")
