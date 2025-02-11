import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path
import math
import os

"""
This script contains the functions to process the video and propose a score for an athlete's climbing performance.
The algorithm detects when holds are gripped by each hand, including duoholds.
It also detects the first frame of a fall based on movement thresholds. 
The algorithm then determines the progression of the hand that moved during the interval of interest.
It also looks at the hip movement before a fall to determine if there is sufficient hip movement.

Input:
- Climbing video
- Landmarks file
- Annotations file

Output:
- Left hand holds
- Right hand holds
- Interval for plus progression
- Hand progression
- Hip progression
"""


def get_total_frames(input_video_path: Path) -> int:
    capture = cv2.VideoCapture(str(input_video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return total_frames


def calculate_movement_over_interval(x_coords, y_coords, interval):
    distances = []
    for i in range(len(x_coords) - interval):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i + interval], y_coords[i + interval]
        distances.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    return np.array(distances)


def detect_fall(input_landmarks_path, distance_threshold=0.15, fall_interval=10, frame_window=200):
    """
    Detects the first frame of a fall based on movement thresholds, NaN checks, and proximity to the end of the video.

    Args:
        input_landmarks_path (str): Path to the parquet file containing landmarks.
        distance_threshold (float): The movement threshold for detecting a fall.
        fall_interval (int): The interval over which movement is measured.
        frame_window (int): The number of frames from the end within which a fall is valid.

    Returns:
        int, int: The frame where the fall starts (fall_frame) and the fall_interval,
                  or (None, None) if no fall is detected.
    """
    # Load landmarks
    landmarks = pd.read_parquet(input_landmarks_path)

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

    # Compute the movement of the hip center over intervals
    hip_middle_movement = calculate_movement_over_interval(hip_middle_x, hip_middle_y, fall_interval)

    # Detect the first frame of fall
    total_frames = len(hip_middle_movement)
    for i in range(total_frames):
        interval_end = i + fall_interval
        if hip_middle_movement[i] > distance_threshold and interval_end <= total_frames:
            # Extract the p-values for the interval and check if any are NaN
            p_values_interval = pd.concat([left_hip_p[i:interval_end], right_hip_p[i:interval_end]])
            # Check if the interval is within the last `frame_window` frames
            if not p_values_interval.isna().any() and interval_end >= total_frames - frame_window:
                # Calculate the fall_frame
                fall_frame = i # The frame when the fall starts
                return fall_frame, fall_interval

    return None, None  # No fall detected


def extract_first_frames(data):
    first_frames = {}

    for frame, hold in data:
        if hold not in first_frames:
            first_frames[hold] = frame

    return first_frames


def process_video_and_track_holds(input_video_path, input_landmarks_path, annotations_path, control_framenumber=70):
    """
    Process the video to detect when holds are gripped by each hand, including duoholds.

    Returns:
        left_hand_holds (dict): {hold_number: first_frame}
        right_hand_holds (dict): {hold_number: first_frame}
    """

    # Load landmarks
    landmarks = pd.read_parquet(input_landmarks_path)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Open video
    capture = cv2.VideoCapture(input_video_path)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tracking how long each hand is on a hold
    left_hold_duration = {}  # {hold_number: frame count}
    right_hold_duration = {}  # {hold_number: frame count}

    # Final hold results
    left_hand_holds = {}  # {hold_number: first_frame}
    right_hand_holds = {}  # {hold_number: first_frame}

    # Tracking first control of duoholds
    duohold_first_control = {}  # {lower_score: 'left' or 'right'}

    # Keep track of last hold each hand was on
    last_left_hold = None
    last_right_hold = None

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        current_landmarks = landmarks.iloc[frame_idx]
        landmarks_x = (current_landmarks[current_landmarks.index.str.contains('_x')] * width).values
        landmarks_y = (current_landmarks[current_landmarks.index.str.contains('_y')] * height).values
        valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

        left_hand, right_hand = None, None

        # Track hand positions
        if valid_landmarks[15]:  # Left wrist
            left_hand = (int(landmarks_x[15]), int(landmarks_y[15]))

        if valid_landmarks[16]:  # Right wrist
            right_hand = (int(landmarks_x[16]), int(landmarks_y[16]))

        # Process holds
        for annotation in annotations:
            center = tuple(annotation['center'])
            radius = annotation['radius']
            hold_number = annotation['number']

            # Convert hold_number to integer for single holds
            if '/' not in hold_number:
                hold_number = int(hold_number)

            left_hand_in_radius = left_hand and math.hypot(left_hand[0] - center[0], left_hand[1] - center[1]) <= radius
            right_hand_in_radius = right_hand and math.hypot(right_hand[0] - center[0], right_hand[1] - center[1]) <= radius

            # Handle duoholds correctly
            if '/' in annotation['number']:
                lower_score, higher_score = map(int, hold_number.split('/'))

                # Initialize tracking if not present
                left_hold_duration.setdefault(lower_score, 0)
                right_hold_duration.setdefault(lower_score, 0)

                # First, track the lower score hold
                if left_hand_in_radius:
                    if last_left_hold is None or last_left_hold <= lower_score:
                        left_hold_duration[lower_score] += 1
                        last_left_hold = lower_score
                elif last_left_hold is not None and last_left_hold > lower_score:
                    left_hold_duration[lower_score] = 0  # Reset only if hand moves up

                if right_hand_in_radius:
                    if last_right_hold is None or last_right_hold <= lower_score:
                        right_hold_duration[lower_score] += 1
                        last_right_hold = lower_score
                elif last_right_hold is not None and last_right_hold > lower_score:
                    right_hold_duration[lower_score] = 0  # Reset only if hand moves up

                # Award lower score to first hand that stays 70 frames
                if lower_score not in left_hand_holds and lower_score not in right_hand_holds:
                    if left_hold_duration[lower_score] >= control_framenumber:
                        left_hand_holds[lower_score] = frame_idx
                        duohold_first_control[lower_score] = 'left'
                    elif right_hold_duration[lower_score] >= control_framenumber:
                        right_hand_holds[lower_score] = frame_idx
                        duohold_first_control[lower_score] = 'right'

                # Award the higher score exactly 70 frames after the first hold was secured
                if lower_score in duohold_first_control:
                    if duohold_first_control[lower_score] == 'left':
                        if right_hand_in_radius:
                            right_hold_duration[higher_score] = right_hold_duration.get(higher_score, 0) + 1
                        else:
                            right_hold_duration[higher_score] = 0  # Reset only if right hand moves away

                        if higher_score not in right_hand_holds and right_hold_duration[higher_score] >= control_framenumber:
                            right_hand_holds[higher_score] = frame_idx

                    elif duohold_first_control[lower_score] == 'right':
                        if left_hand_in_radius:
                            left_hold_duration[higher_score] = left_hold_duration.get(higher_score, 0) + 1
                        else:
                            left_hold_duration[higher_score] = 0  # Reset only if left hand moves away

                        if higher_score not in left_hand_holds and left_hold_duration[higher_score] >= control_framenumber:
                            left_hand_holds[higher_score] = frame_idx

            # Handle single holds
            else:
                left_hold_duration.setdefault(hold_number, 0)
                right_hold_duration.setdefault(hold_number, 0)

                if left_hand_in_radius:
                    if last_left_hold is None or last_left_hold <= hold_number:
                        left_hold_duration[hold_number] += 1
                        last_left_hold = hold_number
                elif last_left_hold is not None and last_left_hold > hold_number:
                    left_hold_duration[hold_number] = 0  # Reset only if hand moves up

                if right_hand_in_radius:
                    if last_right_hold is None or last_right_hold <= hold_number:
                        right_hold_duration[hold_number] += 1
                        last_right_hold = hold_number
                elif last_right_hold is not None and last_right_hold > hold_number:
                    right_hold_duration[hold_number] = 0  # Reset only if hand moves up

                if hold_number not in left_hand_holds and hold_number not in right_hand_holds:
                    if left_hold_duration[hold_number] >= control_framenumber:
                        left_hand_holds[int(hold_number)] = frame_idx
                    elif right_hold_duration[hold_number] >= control_framenumber:
                        right_hand_holds[int(hold_number)] = frame_idx

        frame_idx += 1

    capture.release()

    return left_hand_holds, right_hand_holds


def get_interval_for_plus_progression(left_hand_holds, right_hand_holds, fall_frame, fall_interval, control_framenumber=70):
    """
    Determine the interval for plus progression based on the highest controlled hold.
    """

    # Merge left and right hand holds
    all_holds = {**left_hand_holds, **right_hand_holds}

    # Find the highest numerical hold controlled
    last_controlled_hold = max(all_holds.keys())

    # Get the correct frame number when this hold was first controlled
    check_start_frame = all_holds[last_controlled_hold] - 70  # Adjust based on testing

    # Adjust check_end_frame by subtracting fall_interval
    check_end_frame = fall_frame - 10 # Adjust based on testing

    # Determine the next free hold correctly
    if '/' in str(last_controlled_hold):  # Handle duoholds
        lower_score, higher_score = map(int, str(last_controlled_hold).split('/'))
        next_free_hold = higher_score if lower_score in all_holds else lower_score
    else:
        next_free_hold = last_controlled_hold + 1

    return check_start_frame, check_end_frame, last_controlled_hold, next_free_hold


def get_hand_positions_at_frame(input_landmarks_path, annotations_path, check_start_frame, width, height,
                                last_controlled_hold, lookback_frames=200):
    """
    Get the positions of the hands at a specific frame, prioritizing the immediate frame,
    but looking back if necessary to identify the last known hold.
    """

    # Load landmarks
    landmarks = pd.read_parquet(input_landmarks_path)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Extract annotations as a dictionary
    annotation_holds = {
        annotation['number']: {
            'center': tuple(annotation['center']),
            'radius': annotation['radius']
        }
        for annotation in annotations
    }

    last_known_holds = {'left': None, 'right': None}

    def find_hold(hand_position):
        """Helper to find the correct hold for a given hand position."""
        for hold_number, annotation in annotation_holds.items():
            center = annotation['center']
            radius = annotation['radius']

            if math.hypot(hand_position[0] - center[0], hand_position[1] - center[1]) <= radius:
                return hold_number
        return None

    def resolve_duohold(hold_number):
        """Resolve whether a duohold should return its lower or higher score."""
        if '/' in hold_number:
            lower_score, higher_score = map(int, hold_number.split('/'))

            # If the last controlled hold is higher than the duohold's higher score, assign the higher score
            if last_controlled_hold > higher_score:
                return higher_score

            # Otherwise, assign the lower score
            return lower_score

        return int(hold_number)  # Convert single holds to integer for consistency

    # Check immediate frame at `check_start_frame`
    frame_landmarks = landmarks.iloc[check_start_frame]
    landmarks_x = (frame_landmarks[frame_landmarks.index.str.contains('_x')] * width).values
    landmarks_y = (frame_landmarks[frame_landmarks.index.str.contains('_y')] * height).values
    valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

    # Check left hand (index 15) and right hand (index 16) at `check_start_frame`
    if valid_landmarks[15]:
        left_hand = (int(landmarks_x[15]), int(landmarks_y[15]))
        hold = find_hold(left_hand)
        if hold:
            last_known_holds['left'] = resolve_duohold(hold)

    if valid_landmarks[16]:
        right_hand = (int(landmarks_x[16]), int(landmarks_y[16]))
        hold = find_hold(right_hand)
        if hold:
            last_known_holds['right'] = resolve_duohold(hold)

    # If both hands have valid holds, return immediately
    if last_known_holds['left'] is not None and last_known_holds['right'] is not None:
        return last_known_holds['left'], last_known_holds['right']

    # If not, look back over `lookback_frames`
    for frame_idx in range(max(0, check_start_frame - lookback_frames), check_start_frame):
        frame_landmarks = landmarks.iloc[frame_idx]
        landmarks_x = (frame_landmarks[frame_landmarks.index.str.contains('_x')] * width).values
        landmarks_y = (frame_landmarks[frame_landmarks.index.str.contains('_y')] * height).values
        valid_landmarks = ~np.isnan(landmarks_x) & ~np.isnan(landmarks_y)

        if last_known_holds['left'] is None and valid_landmarks[15]:
            left_hand = (int(landmarks_x[15]), int(landmarks_y[15]))
            hold = find_hold(left_hand)
            if hold:
                last_known_holds['left'] = resolve_duohold(hold)

        if last_known_holds['right'] is None and valid_landmarks[16]:
            right_hand = (int(landmarks_x[16]), int(landmarks_y[16]))
            hold = find_hold(right_hand)
            if hold:
                last_known_holds['right'] = resolve_duohold(hold)

        # If both hands now have valid holds, we can stop early
        if last_known_holds['left'] is not None and last_known_holds['right'] is not None:
            break

    return last_known_holds['left'], last_known_holds['right']


def check_hand_movement_in_interval(input_landmarks_path, check_start_frame, check_end_frame, width, height):
    """
    Check which hand is moving more in the interval of interest.
    Returns True for the hand with the greater movement, and False for the other.
    """
    # Load landmarks
    landmarks = pd.read_parquet(input_landmarks_path)

    # Initialize variables to store the positions of the hands at start and end frames
    start_left_hand = None
    start_right_hand = None
    end_left_hand = None
    end_right_hand = None

    # Function to get hand position, checking the validity of landmarks
    def get_hand_position(frame_idx, hand_index):
        frame_landmarks = landmarks.iloc[frame_idx]
        landmarks_x = (frame_landmarks[frame_landmarks.index.str.contains('_x')] * width).values
        landmarks_y = (frame_landmarks[frame_landmarks.index.str.contains('_y')] * height).values
        if not np.isnan(landmarks_x[hand_index]) and not np.isnan(landmarks_y[hand_index]):
            return (int(landmarks_x[hand_index]), int(landmarks_y[hand_index]))
        return None

    # Find the first valid hand position at start frame (check_start_frame)
    start_left_hand = get_hand_position(check_start_frame, 15)  # Left wrist index
    start_right_hand = get_hand_position(check_start_frame, 16)  # Right wrist index

    # If the start frame has no valid left hand, look for the next valid frame
    if start_left_hand is None:
        for frame_idx in range(check_start_frame + 1, check_end_frame):
            start_left_hand = get_hand_position(frame_idx, 15)
            if start_left_hand is not None:
                break

    # If the start frame has no valid right hand, look for the next valid frame
    if start_right_hand is None:
        for frame_idx in range(check_start_frame + 1, check_end_frame):
            start_right_hand = get_hand_position(frame_idx, 16)
            if start_right_hand is not None:
                break

    # Find the first valid hand position at end frame (check_end_frame)
    end_left_hand = get_hand_position(check_end_frame, 15)  # Left wrist index
    end_right_hand = get_hand_position(check_end_frame, 16)  # Right wrist index

    # If the end frame has no valid left hand, look for the previous valid frame
    if end_left_hand is None:
        for frame_idx in range(check_end_frame - 1, check_start_frame, -1):
            end_left_hand = get_hand_position(frame_idx, 15)
            if end_left_hand is not None:
                break

    # If the end frame has no valid right hand, look for the previous valid frame
    if end_right_hand is None:
        for frame_idx in range(check_end_frame - 1, check_start_frame, -1):
            end_right_hand = get_hand_position(frame_idx, 16)
            if end_right_hand is not None:
                break

    # Check if both hands are still None after looking for alternatives
    if start_left_hand is None or start_right_hand is None or end_left_hand is None or end_right_hand is None:
        return False, False  # Return False if any hand is missing

    # Calculate the total distance moved for both hands
    left_hand_distance = 0
    right_hand_distance = 0

    if start_left_hand and end_left_hand:
        left_hand_distance = math.hypot(end_left_hand[0] - start_left_hand[0], end_left_hand[1] - start_left_hand[1])
        print(f"Left hand moved: {left_hand_distance} pixels")

    if start_right_hand and end_right_hand:
        right_hand_distance = math.hypot(end_right_hand[0] - start_right_hand[0], end_right_hand[1] - start_right_hand[1])
        print(f"Right hand moved: {right_hand_distance} pixels")

    # Compare distances to determine which hand moved more
    if left_hand_distance > right_hand_distance:
        return True, False  # Left hand moved more
    elif right_hand_distance > left_hand_distance:
        return False, True  # Right hand moved more
    else:
        return False, False  # No significant movement or equal movement


def determine_hand_progression(
    annotations_path,
    input_landmarks_path,
    next_free_hold,
    left_hand_hold,
    right_hand_hold,
    left_moved,
    right_moved,
    check_start_frame,
    check_end_frame,
    width,
    height
):
    """
    Determine the progression of the hand that moved during the interval of interest.
    Calculates the greater distance (horizontal or vertical) between the hold of the moving hand
    and the next free hold. Also checks if the moving hand has progressed more than half of the
    greater distance in x or y direction during the interval.

    Args:
        annotations_path (Path): Path to the annotations file (JSON).
        input_landmarks_path (Path): Path to the landmarks file (Parquet).
        next_free_hold (int): The next free hold number.
        left_hand_hold (int): The hold number where the left hand is.
        right_hand_hold (int): The hold number where the right hand is.
        left_moved (bool): Whether the left hand moved more.
        right_moved (bool): Whether the right hand moved more.
        check_start_frame (int): Start frame of the interval of interest.
        check_end_frame (int): End frame of the interval of interest.
        width (int): Width of the frame for scaling landmarks.
        height (int): Height of the frame for scaling landmarks.

    Returns:
        dict: A dictionary containing:
            - greater_distance: {"axis": "x" or "y", "value": distance}
            - hand_progression: True if the moving hand moved more than half of the greater distance, else False.
    """

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Load landmarks
    landmarks = pd.read_parquet(input_landmarks_path)

    # Preprocess annotations to handle duoholds properly
    hold_positions = {}  # {hold_number: (x, y)}
    for annotation in annotations:
        hold_number = annotation['number']
        center = tuple(annotation['center'])  # (x, y) position

        if '/' in hold_number:
            lower_score, higher_score = map(int, hold_number.split('/'))
            hold_positions[lower_score] = center  # Assign both scores the same position
            hold_positions[higher_score] = center
        else:
            hold_positions[int(hold_number)] = center  # Store normal holds as is

    # Helper function to get the center of a hold
    def get_hold_center(hold_number):
        """Retrieve the center of a hold, correctly handling duoholds."""
        return hold_positions.get(hold_number, None)

    # Helper function to get hand position
    def get_hand_position(frame_idx, hand_index):
        frame_landmarks = landmarks.iloc[frame_idx]
        landmarks_x = (frame_landmarks[frame_landmarks.index.str.contains('_x')] * width).values
        landmarks_y = (frame_landmarks[frame_landmarks.index.str.contains('_y')] * height).values
        if not np.isnan(landmarks_x[hand_index]) and not np.isnan(landmarks_y[hand_index]):
            return (int(landmarks_x[hand_index]), int(landmarks_y[hand_index]))
        return None

    # Determine which hand moved and get its hold
    if left_moved:
        moving_hand_hold = left_hand_hold
        hand_index = 15  # Left wrist index
    elif right_moved:
        moving_hand_hold = right_hand_hold
        hand_index = 16  # Right wrist index
    else:
        raise ValueError("No hand movement detected.")

    # Get the centers of the moving hand's hold and the next free hold
    moving_hand_center = get_hold_center(moving_hand_hold)
    next_free_hold_center = get_hold_center(next_free_hold)

    if not moving_hand_center or not next_free_hold_center:
        raise ValueError(f"Invalid hold numbers or missing annotation centers: {moving_hand_hold}, {next_free_hold}")

    # Calculate horizontal (x) and vertical (y) distances
    x_distance = abs(next_free_hold_center[0] - moving_hand_center[0])
    y_distance = abs(next_free_hold_center[1] - moving_hand_center[1])

    # Determine the greater distance
    if x_distance > y_distance:
        greater_distance = {"axis": "x", "value": x_distance}
        threshold_distance = x_distance * 0.45 # this can be adjusted ---> determines how much distance needs to be covered for a +
    else:
        greater_distance = {"axis": "y", "value": y_distance}
        threshold_distance = y_distance * 0.45

    # Get the starting and ending positions of the moving hand
    start_hand_position = None
    end_hand_position = None

    # Find valid starting position
    for frame_idx in range(check_start_frame, check_end_frame + 1):
        start_hand_position = get_hand_position(frame_idx, hand_index)
        if start_hand_position:
            break

    # Find valid ending position
    for frame_idx in range(check_end_frame, check_start_frame - 1, -1):
        end_hand_position = get_hand_position(frame_idx, hand_index)
        if end_hand_position:
            break

    if not start_hand_position or not end_hand_position:
        raise ValueError("Invalid hand positions in the interval of interest.")

    # Calculate movement in x and y directions
    hand_movement_x = abs(end_hand_position[0] - start_hand_position[0])
    hand_movement_y = abs(end_hand_position[1] - start_hand_position[1])

    # Check if hand progression exceeds half the greater distance
    if greater_distance["axis"] == "x":
        hand_progression = hand_movement_x > threshold_distance
    else:
        hand_progression = hand_movement_y > threshold_distance

    return {
        "greater_distance": greater_distance,
        "hand_progression": hand_progression,
    }


def calculate_hip_progression(input_landmarks_path, start_frame, end_frame, width, height, interval=10,
                              threshold=1.5e-05):
    """
    Determine if there is sufficient hip movement in the interval before a fall.

    Args:
        landmarks (pd.DataFrame): DataFrame containing hip coordinates ('left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y').
        start_frame (int): Start frame of the interval of interest.
        end_frame (int): End frame of the interval of interest.
        width (int): Frame width in pixels (used for normalization).
        height (int): Frame height in pixels (used for normalization).
        interval (int): Number of frames over which to calculate movement (default is 10).
        threshold (float): Normalized Euclidean distance threshold to consider movement sufficient.

    Returns:
        dict: Contains:
            - 'left_hip_progression': True if left hip movement exceeds threshold, False otherwise.
            - 'right_hip_progression': True if right hip movement exceeds threshold, False otherwise.
    """

    landmarks = pd.read_parquet(input_landmarks_path)

    # Normalize coordinates
    left_hip_x = landmarks['left_hip_x'].values / width
    left_hip_y = landmarks['left_hip_y'].values / height
    right_hip_x = landmarks['right_hip_x'].values / width
    right_hip_y = landmarks['right_hip_y'].values / height

    # Calculate Euclidean distances over intervals
    def calculate_movement(x_coords, y_coords, interval):
        distances = []
        for i in range(len(x_coords) - interval):
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i + interval], y_coords[i + interval]
            distances.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        return np.array(distances)

    # Select the range of interest
    range_indices = range(start_frame, end_frame + 1 - interval)

    # Check the movement of the left and right hips
    left_hip_movement = calculate_movement(left_hip_x[range_indices], left_hip_y[range_indices], interval)
    right_hip_movement = calculate_movement(right_hip_x[range_indices], right_hip_y[range_indices], interval)

    # Check if movement exceeds threshold
    left_hip_progression = np.any(left_hip_movement > threshold)
    right_hip_progression = np.any(right_hip_movement > threshold)

    return {
        "left_hip_progression": left_hip_progression,
        "right_hip_progression": right_hip_progression,
    }


def main():
    file_name = 'edited_villars_men_semifinals_n110_plus_14+'

    # Automatically generated paths based on the file name
    input_video_path = f'../data/input/videos/{file_name}.mp4'
    input_landmarks_path = f'../data/input/landmarks/{file_name}_coordinates_local.parquet'
    annotations_path = f'../data/input/topos/{file_name}_annotations.json'
    dataframe_path = '../data/output/plus_algorithm/results_algorithm_performance.pd'

    # -------------------------------------------------------------------------------------

    # Get the frame dimensions
    capture = cv2.VideoCapture(input_video_path)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    # Print out total number of frames of the input video
    total_frames = get_total_frames(input_video_path)
    print(f"Total number of frames: {total_frames}")

    # Detect holds separately for left and right hands
    left_hand_holds, right_hand_holds = process_video_and_track_holds(input_video_path, input_landmarks_path,
                                                                      annotations_path)

    # Print frame numbers for left and right hand holds
    print("\nLeft Hand Holds (Frame Number - Hold Number):")
    for hold, frame in left_hand_holds.items():
        print(f"Frame {frame} - Hold {hold}")

    print("\nRight Hand Holds (Frame Number - Hold Number):")
    for hold, frame in right_hand_holds.items():
        print(f"Frame {frame} - Hold {hold}")

    # Print out at what frame number the fall was detected and give the interval of interest
    fall_frame, fall_interval = detect_fall(input_landmarks_path)

    if fall_frame is not None:
        print(f"Fall detected starting at frame: {fall_frame}")

        # Get progression interval based on the last controlled hold
        check_start_frame, check_end_frame, last_controlled_hold, next_free_hold = get_interval_for_plus_progression(
            left_hand_holds, right_hand_holds, fall_frame, fall_interval
        )

        print(f"Interval for plus progression: {check_start_frame} - {check_end_frame}")
        print(f"Last controlled hold: {last_controlled_hold}, Next free hold: {next_free_hold}")

        # Call check_hand_movement_in_interval function with the interval of interest
        left_moved, right_moved = check_hand_movement_in_interval(input_landmarks_path, check_start_frame,
                                                                  check_end_frame, width, height)
        print(f"Left hand moved more: {left_moved}")
        print(f"Right hand moved more: {right_moved}")

        # Get the last holds gripped for each hand before the fall
        left_hand_hold, right_hand_hold = get_hand_positions_at_frame(
            input_landmarks_path=input_landmarks_path,
            annotations_path=annotations_path,
            check_start_frame=check_start_frame,
            width=width,
            height=height,
            last_controlled_hold=last_controlled_hold,  # Pass last controlled hold here!
            lookback_frames=200  # Adjust as needed
        )

        print(f"Left hand at hold: {left_hand_hold}")
        print(f"Right hand at hold: {right_hand_hold}")

        # Determine hand progression
        hand_progression = determine_hand_progression(
            annotations_path, input_landmarks_path, next_free_hold, left_hand_hold, right_hand_hold,
            left_moved, right_moved, check_start_frame, check_end_frame, width, height
        )
        print(f"Greater distance: {hand_progression['greater_distance']}")
        print(f"Hand progression: {'Yes' if hand_progression['hand_progression'] else 'No'}")

        # Calculate hip progression
        hip_progression = calculate_hip_progression(
            input_landmarks_path, check_start_frame, check_end_frame, width, height, interval=10, threshold=1.5e-05
        )
        print(f"Left Hip Progression: {'Yes' if hip_progression['left_hip_progression'] else 'No'}")
        print(f"Right Hip Progression: {'Yes' if hip_progression['right_hip_progression'] else 'No'}")

        # Final decision: Determine if it was a "plus"
        is_plus = (
                hand_progression['hand_progression'] and
                (hip_progression['left_hip_progression'] or hip_progression['right_hip_progression'])
        )

        if is_plus:
            print("plus")
            print(f"Final score: {last_controlled_hold}+")
            final_score = f"{last_controlled_hold}+"
        else:
            print("no plus")
            print(f"Final score: {last_controlled_hold}")
            final_score = int(last_controlled_hold)

    else:
        print("No fall detected.")

    if fall_frame is not None:
        # Collect results in a dictionary
        results = {
            "video_name": file_name,
            "fall_frame": fall_frame,
            "last_controlled_hold": last_controlled_hold,
            "next_free_hold": next_free_hold,
            "hand_progression": hand_progression['hand_progression'],
            "hip_progression": (hip_progression['left_hip_progression'] or hip_progression['right_hip_progression']),
            "plus": is_plus,
            "final_score": final_score
        }

        """
        # Create or load the DataFrame
        if os.path.exists(dataframe_path):
            plus_algorithm_results = pd.read_pickle(dataframe_path)
        else:
            plus_algorithm_results = pd.DataFrame(columns=["video_name", "fall_frame", "last_controlled_hold",
                                                           "next_free_hold", "hand_progression", "hip_progression",
                                                           "plus"])

        # Update or add the result
        if file_name in plus_algorithm_results["video_name"].values:
            # Update the existing row
            for key, value in results.items():
                plus_algorithm_results.loc[plus_algorithm_results["video_name"] == file_name, key] = value
        else:
            # Add a new row
            plus_algorithm_results = pd.concat([plus_algorithm_results, pd.DataFrame([results])], ignore_index=True)

        # Save the updated DataFrame
        plus_algorithm_results.to_pickle(dataframe_path)

        # Print confirmation
        print(f"Results saved to {dataframe_path}")
        """


if __name__ == "__main__":
    main()
