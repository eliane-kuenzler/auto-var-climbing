import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

"""
This script generates an output video with scoring logic for climbing competitions.
- Landmark Tracking: The script tracks the hips and hands of the climber.
- Scoring Logic: It calculates the score based on the climber's interaction with climbing holds.
- Duohold Detection: It distinguishes between single and duoholds, requiring both hands to engage duoholds.

Input:
- Climbing video
- Landmarks data
- Annotated topo

Output:
- Video with scoring logic overlay
"""

# Control variable for the number of frames to consider the hold as controlled
control_framenumber = 70

# Paths
file_name = 'edited_villars_men_semifinals_n110_plus_14+'

# Automatically generated paths based on the file name
input_video_path = f'../data/input/videos/{file_name}.mp4'
input_landmarks_path = f'../data/input/landmarks/{file_name}_coordinates_local.parquet'
annotations_path = f'../data/input/topos/{file_name}_annotations.json'
output_video_path = f'../data/output/scoring_algorithm/{file_name}_scoring.mp4'

# Define annotation color
color = (255, 255, 255)  # White for black change to (0, 0, 0)

# Load landmarks
landmarks = pd.read_parquet(input_landmarks_path)

# Load the saved annotations
with open(str(annotations_path), 'r') as f:
    annotations = json.load(f)

# Open the input video
capture = cv2.VideoCapture(str(input_video_path))
ret, frame = capture.read()

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
height, width, _ = frame.shape
output_video = cv2.VideoWriter(
    str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# BlazePose keypoint indices
keypoint_dict = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

# MediaPipe skeleton connections
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

# Initialize hold tracking
hold_frame_counts = {annotation['number']: 0 for annotation in annotations}
hold_gripped = {
    annotation['number']: [False, False] if '/' in annotation['number'] else False
    for annotation in annotations
}

# Initialize tracks
hip_track = []
left_hand_track = []
right_hand_track = []

# Process each frame
for frame_idx in tqdm(range(len(landmarks))):
    ret, frame = capture.read()
    if not ret:
        break

    # Initialize variables for the current frame
    left_hand = None
    right_hand = None

    current_landmarks = landmarks.iloc[frame_idx]

    # Scale landmarks to pixel coordinates
    landmarks_x = (current_landmarks[current_landmarks.index.str.contains('_x')] * width).values
    landmarks_y = (current_landmarks[current_landmarks.index.str.contains('_y')] * height).values
    valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

    # Draw skeleton connections
    for start_idx, end_idx in connections:
        if valid_landmarks[start_idx] and valid_landmarks[end_idx]:
            start_point = (int(landmarks_x[start_idx]), int(landmarks_y[start_idx]))
            end_point = (int(landmarks_x[end_idx]), int(landmarks_y[end_idx]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

    # Track hips and hands
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

    # Draw tracks (hip and hands) with distinct colors
    for i in range(1, len(hip_track)):
        adjusted_hip = (hip_track[i][0], hip_track[i][1])
        cv2.line(frame, adjusted_hip, adjusted_hip, (0, 0, 255), 2)  # Red for hip center

    for i in range(1, len(left_hand_track)):
        adjusted_left_hand = (left_hand_track[i][0], left_hand_track[i][1])
        cv2.line(frame, adjusted_left_hand, adjusted_left_hand, (255, 0, 0), 2)  # Blue for left hand

    for i in range(1, len(right_hand_track)):
        adjusted_right_hand = (right_hand_track[i][0], right_hand_track[i][1])
        cv2.line(frame, adjusted_right_hand, adjusted_right_hand, (0, 255, 0), 2)  # Green for right hand

# ------------------------------------ S C O R I N G    L O G I C ----------------------------------------------

    # Process annotations
    for annotation in annotations:
        center = tuple(annotation['center'])
        radius = annotation['radius']
        hold_number = annotation['number']

        # Draw annotation
        cv2.circle(frame, center, radius, color, 2)
        cv2.circle(frame, center, 2, color, -1)

        # Calculate text position (outside the circle)
        angle = math.atan2(-20, 20)  # Example angle for offset direction
        line_end_x = int(center[0] + radius * math.cos(angle))
        line_end_y = int(center[1] + radius * math.sin(angle))
        text_position = (line_end_x + 10, line_end_y - 10)  # Slight offset for text placement

        # Draw line from circle boundary to text
        cv2.line(frame, (line_end_x, line_end_y), text_position, color, 2)

        # Give out hold number on frame
        cv2.putText(frame, hold_number, (center[0] + radius, center[1] - radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Evaluate hold interaction
        if '/' in hold_number:  # Duohold
            scores = hold_number.split('/')
            lower_score, higher_score = int(scores[0]), int(scores[1])

            # Check if the left hand is within the radius of the hold
            if left_hand and math.hypot(left_hand[0] - center[0], left_hand[1] - center[1]) <= radius:
                hold_gripped[hold_number][0] = True

            # Check if the right hand is within the radius of the hold
            if right_hand and math.hypot(right_hand[0] - center[0], right_hand[1] - center[1]) <= radius:
                hold_gripped[hold_number][1] = True

            # Increment frame counts only if at least one hand is within the radius
            if hold_gripped[hold_number][0] or hold_gripped[hold_number][1]:
                hold_frame_counts[hold_number] += 1

            # Update duohold scoring logic
            if hold_frame_counts[hold_number] >= control_framenumber:
                # For the higher score (e.g., 32 in "31/32"), both hands must be on the hold for the required frame count
                if hold_gripped[hold_number][0] and hold_gripped[hold_number][1]:
                    hold_gripped[hold_number] = [True, True]
                else:
                    # If only one hand is on the hold, it qualifies for the lower score (e.g., 31 in "31/32")
                    hold_gripped[hold_number] = [True, False]

        else:  # Single hold
            # Evaluate single holds as before
            if (left_hand and math.hypot(left_hand[0] - center[0], left_hand[1] - center[1]) <= radius) or \
                    (right_hand and math.hypot(right_hand[0] - center[0], right_hand[1] - center[1]) <= radius):
                hold_frame_counts[hold_number] += 1
                if hold_frame_counts[hold_number] >= control_framenumber:
                    hold_gripped[hold_number] = True

    # Calculate current score
    all_scores = []
    for hold_number, gripped in hold_gripped.items():
        if isinstance(gripped, list):  # Duohold
            if gripped[1]:
                all_scores.append(int(hold_number.split('/')[1]))
            elif gripped[0]:
                all_scores.append(int(hold_number.split('/')[0]))
        else:
            if gripped:
                all_scores.append(int(hold_number))

    current_score = max(all_scores, default='-')
    cv2.putText(frame, f"Score: {current_score}", (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write frame to output video
    output_video.write(frame)

# Release resources
capture.release()
output_video.release()
print("Video generation complete.")
