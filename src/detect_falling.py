import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
This script allows you to detect falling in climbing videos.
It uses the hip middle point movement over intervals to detect falling.

Input:
- Climbing video

Output:
- Video with falling detection status
"""

file_name = "edited_villars_men_semifinals_n110_plus_14+"

# Automatically generated paths based on the file name
input_video_path = f'../data/input/videos/{file_name}.mp4'
input_landmarks_path = f'../data/input/landmarks/{file_name}_coordinates_local.parquet'
output_video_path = f'../data/output/detect_falling/{file_name}_detect_falling.mp4'

# BlazePose keypoint indices
keypoint_dict = {
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

# Threshold and interval for fall detection
distance_threshold = 0.15
interval = 10

# Load landmarks
landmarks = pd.read_parquet(input_landmarks_path)

# Open the input video
capture = cv2.VideoCapture(str(input_video_path))
ret, frame = capture.read()

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
height, width, _ = frame.shape
output_video = cv2.VideoWriter(
    str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Extract hip coordinates and p-values
left_hip_x = landmarks['left_hip_x']
left_hip_y = landmarks['left_hip_y']
right_hip_x = landmarks['right_hip_x']
right_hip_y = landmarks['right_hip_y']
left_hip_p = landmarks['left_hip_p']
right_hip_p = landmarks['right_hip_p']

# Calculate the hip middle point
hip_middle_x = (left_hip_x + right_hip_x) / 2
hip_middle_y = (left_hip_y + right_hip_y) / 2

# Compute Euclidean distance of the hip middle point over intervals
def calculate_movement_over_interval(x_coords, y_coords, interval):
    distances = []
    for i in range(len(x_coords) - interval):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i + interval], y_coords[i + interval]
        distances.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return np.array(distances)

hip_middle_movement = calculate_movement_over_interval(hip_middle_x, hip_middle_y, interval)

# Status tracking
status = ["climbing"] * len(landmarks)
falling_detected = False  # Track if falling has been detected

for i in range(len(hip_middle_movement)):
    # Check if the movement is above the threshold and if there are no NaN p-values in the interval
    interval_end = i + interval
    if hip_middle_movement[i] > distance_threshold and i + interval <= len(status):
        # Extract the p-values for the interval and check if any are NaN
        p_values_interval = pd.concat([left_hip_p[i:interval_end], right_hip_p[i:interval_end]])
        # Check if the interval is within the last 200 frames
        if not p_values_interval.isna().any() and interval_end >= len(status) - 200:
            falling_detected = True
            # Set all remaining frames to "falling"
            status[i:] = ["falling"] * (len(status) - i)
            break  # No need to continue looping once falling is detected

# the following code is for the video output
# Process each frame
for frame_idx in tqdm(range(len(landmarks))):
    ret, frame = capture.read()
    if not ret:
        break

    # Draw skeleton connections
    current_landmarks = landmarks.iloc[frame_idx]

    # Scale landmarks to pixel coordinates
    landmarks_x = (current_landmarks[current_landmarks.index.str.contains('_x')] * width).values
    landmarks_y = (current_landmarks[current_landmarks.index.str.contains('_y')] * height).values
    valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

    # Draw skeleton connections in white
    for start_idx, end_idx in connections:
        if valid_landmarks[start_idx] and valid_landmarks[end_idx]:
            start_point = (int(landmarks_x[start_idx]), int(landmarks_y[start_idx]))
            end_point = (int(landmarks_x[end_idx]), int(landmarks_y[end_idx]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # White for skeleton

    # Display climbing or falling status
    current_status = status[frame_idx]

    # Dynamically set text position and size based on video dimensions
    text_position = (int(0.1 * width), int(0.1 * height))  # 10% from the left and 10% from the top
    font_scale = 1.5 #width / 2000 --> Scale font size relative to video width
    font_thickness = 3 #max(1, int(width / 800)) --> Adjust thickness based on video width

    # Set text color based on the status: (0, 0, 0) black and (255, 255, 255) white
    text_color = (0, 0, 0) if current_status == "climbing" else (0, 0, 255)

    # Render the status text
    cv2.putText(
        frame,
        f"Status: {current_status}",
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness
    )

    # Write the frame to the output video
    output_video.write(frame)

# Release resources
capture.release()
output_video.release()
print("Fall detection video generated successfully.")
