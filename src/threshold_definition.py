import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path
import math
import os
import re

"""
This scrip is here to test all the different variables and set the right thresholds for the plus algorithm.
Specifically the following variables:
- control framenumber for controlling a hold and earning a score
- hip movement threshold for gaining a plus
- distance one hand has to cover to gain a plus

Input:
- Climbing video
- Landmarks file
- Annotations file

Output:
- result dataframe for the different threshold testing
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
                fall_frame = i  # The frame when the fall starts
                return fall_frame, fall_interval

    return None, None  # No fall detected


def extract_first_frames(data):
    first_frames = {}

    for frame, hold in data:
        if hold not in first_frames:
            first_frames[hold] = frame

    return first_frames


def process_video_and_track_holds(input_video_path, input_landmarks_path, annotations_path, control_threshold):
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
            right_hand_in_radius = right_hand and math.hypot(right_hand[0] - center[0],
                                                             right_hand[1] - center[1]) <= radius

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
                    if left_hold_duration[lower_score] >= control_threshold:
                        left_hand_holds[lower_score] = frame_idx
                        duohold_first_control[lower_score] = 'left'
                    elif right_hold_duration[lower_score] >= control_threshold:
                        right_hand_holds[lower_score] = frame_idx
                        duohold_first_control[lower_score] = 'right'

                # Award the higher score exactly 70 frames after the first hold was secured
                if lower_score in duohold_first_control:
                    if duohold_first_control[lower_score] == 'left':
                        if right_hand_in_radius:
                            right_hold_duration[higher_score] = right_hold_duration.get(higher_score, 0) + 1
                        else:
                            right_hold_duration[higher_score] = 0  # Reset only if right hand moves away

                        if higher_score not in right_hand_holds and right_hold_duration[
                            higher_score] >= control_threshold:
                            right_hand_holds[higher_score] = frame_idx

                    elif duohold_first_control[lower_score] == 'right':
                        if left_hand_in_radius:
                            left_hold_duration[higher_score] = left_hold_duration.get(higher_score, 0) + 1
                        else:
                            left_hold_duration[higher_score] = 0  # Reset only if left hand moves away

                        if higher_score not in left_hand_holds and left_hold_duration[
                            higher_score] >= control_threshold:
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
                    if left_hold_duration[hold_number] >= control_threshold:
                        left_hand_holds[int(hold_number)] = frame_idx
                    elif right_hold_duration[hold_number] >= control_threshold:
                        right_hand_holds[int(hold_number)] = frame_idx

        frame_idx += 1

    capture.release()

    return left_hand_holds, right_hand_holds


def get_interval_for_plus_progression(left_hand_holds, right_hand_holds, fall_frame, fall_interval):
    """
    Determine the interval for plus progression based on the highest controlled hold.
    """

    # Merge left and right hand holds
    all_holds = {**left_hand_holds, **right_hand_holds}

    # Find the highest numerical hold controlled
    last_controlled_hold = max(all_holds.keys())

    # Get the correct frame number when this hold was first controlled
    check_start_frame = all_holds[last_controlled_hold] - 30  # Adjust based on testing

    # Adjust check_end_frame by subtracting fall_interval
    check_end_frame = fall_frame - fall_interval  # Adjust based on testing

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
        right_hand_distance = math.hypot(end_right_hand[0] - start_right_hand[0],
                                         end_right_hand[1] - start_right_hand[1])
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
        height,
        hand_progression_threshold  # Pass as an argument
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
        hand_progression_threshold (float): The threshold for determining hand progression.

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
        threshold_distance = x_distance * hand_progression_threshold  # Adjusted for input threshold
    else:
        greater_distance = {"axis": "y", "value": y_distance}
        threshold_distance = y_distance * hand_progression_threshold

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


def calculate_hip_progression(input_landmarks_path, start_frame, end_frame, width, height, hip_threshold, interval=10):
    """
    Determine if there is sufficient hip movement in the interval before a fall.

    Args:
        input_landmarks_path (str): Path to the parquet file containing landmarks.
        start_frame (int): Start frame of the interval of interest.
        end_frame (int): End frame of the interval of interest.
        width (int): Frame width in pixels (used for scaling).
        height (int): Frame height in pixels (used for scaling).
        hip_threshold (int): The threshold value to test for hip progression.
        interval (int): Number of frames over which to calculate movement (default is 10).

    Returns:
        dict: Contains:
            - 'left_hip_progression': True if left hip movement exceeds threshold, False otherwise.
            - 'right_hip_progression': True if right hip movement exceeds threshold, False otherwise.
    """

    landmarks = pd.read_parquet(input_landmarks_path)

    # Convert BlazePose normalized coordinates to pixel values
    left_hip_x = landmarks['left_hip_x'].values * width
    left_hip_y = landmarks['left_hip_y'].values * height
    right_hip_x = landmarks['right_hip_x'].values * width
    right_hip_y = landmarks['right_hip_y'].values * height

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

    # Check if movement exceeds the given threshold
    left_hip_progression = np.any(left_hip_movement > hip_threshold)
    right_hip_progression = np.any(right_hip_movement > hip_threshold)

    return {
        "left_hip_progression": left_hip_progression,
        "right_hip_progression": right_hip_progression,
    }


def hip_progression_tuning():
    """Tunes the hip progression threshold and evaluates accuracy."""

    # Set folder paths
    video_folder = Path("../data/input/videos/")
    landmarks_folder = Path("../data/input/landmarks/")
    annotations_folder = Path("../data/input/topos/")

    # set name of dataframe accordingly to the data sample
    output_dataframe_path = Path("../data/output/plus_algorithm/hip_threshold_tuning_plus_data.pd")  # Save as .pd

    # Set hip progression thresholds to test
    hip_threshold_values = list(range(10, 41))  # From 10 to 40, inclusive

    # Function to extract details from the filename
    def extract_details(video_name):
        parts = video_name.split("_")

        competition = "lenzburg" if "lenzburg" in parts else "villars" if "villars" in parts else None
        gender = "male" if "men" in parts else "female" if "women" in parts else None

        athlete_number_match = re.search(r'n(\d+)', video_name)
        athlete_number = athlete_number_match.group(1) if athlete_number_match else None

        ground_truth_match = re.search(r'(\d+\+?|\d+)', parts[-1])
        ground_truth = ground_truth_match.group(1) if ground_truth_match else None

        return competition, athlete_number, gender, ground_truth

    # Initialize results list
    results = []

    # Iterate over all videos in the folder
    for video_path in video_folder.glob("*.mp4"):
        file_name = video_path.stem  # Get the filename without extension
        print(f'Processing video: {file_name}')

        # Generate corresponding file paths
        input_landmarks_path = landmarks_folder / f"{file_name}_coordinates_local.parquet"
        annotations_path = annotations_folder / f"{file_name}_annotations.json"

        # Get video frame size
        capture = cv2.VideoCapture(str(video_path))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        # Get filename details
        competition, athlete_number, gender, ground_truth = extract_details(file_name)
        ground_truth_plus = "+" in str(ground_truth)

        # Detect holds separately for left and right hands
        left_hand_holds, right_hand_holds = process_video_and_track_holds(video_path, input_landmarks_path,
                                                                          annotations_path, control_threshold=60)

        # Detect fall
        fall_frame, fall_interval = detect_fall(input_landmarks_path)

        if fall_frame is not None:
            # Determine interval for plus progression
            check_start_frame, check_end_frame, last_controlled_hold, next_free_hold = get_interval_for_plus_progression(
                left_hand_holds, right_hand_holds, fall_frame, fall_interval
            )

            # Loop through different hip progression thresholds
            for hip_threshold in hip_threshold_values:
                print(f'Testing hip threshold: {hip_threshold}')

                # Calculate hip progression with the current threshold
                hip_progression = calculate_hip_progression(
                    input_landmarks_path, check_start_frame, check_end_frame, width, height, hip_threshold, interval=10
                )

                # Store the results
                results.append({
                    "file_name": file_name,
                    "hip_progression_threshold": hip_threshold,
                    "hip_progression": hip_progression["left_hip_progression"] or hip_progression[
                        "right_hip_progression"],
                    "ground_truth_score": ground_truth,
                    "ground_truth_plus": ground_truth_plus
                })

    # Convert results to DataFrame
    plus_algorithm_results = pd.DataFrame(results)

    # Save the DataFrame as a .pd file
    plus_algorithm_results.to_pickle(output_dataframe_path)

    print(f"Results saved to {output_dataframe_path}")


def hand_progression_tuning():
    """Tunes the hand progression threshold and evaluates accuracy."""

    # Set folder paths
    video_folder = Path("../data/input/videos/")
    landmarks_folder = Path("../data/input/landmarks/")
    annotations_folder = Path("../data/input/topos/")

    # Set name of dataframe accordingly to the data sample
    output_dataframe_path = Path("../data/output/plus_algorithm/hand_threshold_tuning.pd")  # Save as .pd

    # Set hand progression thresholds to test (from 0.1 to 1.0 in steps of 0.1)
    hand_threshold_values = [x / 10 for x in range(1, 11)]  # Generates [0.1, 0.2, ..., 1.0]

    # Function to extract details from the filename
    def extract_details(video_name):
        parts = video_name.split("_")

        competition = "lenzburg" if "lenzburg" in parts else "villars" if "villars" in parts else None
        gender = "male" if "men" in parts else "female" if "women" in parts else None

        athlete_number_match = re.search(r'n(\d+)', video_name)
        athlete_number = athlete_number_match.group(1) if athlete_number_match else None

        ground_truth_match = re.search(r'(\d+\+?|\d+)', parts[-1])
        ground_truth = ground_truth_match.group(1) if ground_truth_match else None

        return competition, athlete_number, gender, ground_truth

    # Initialize results list
    results = []

    # Iterate over all videos in the folder
    for video_path in video_folder.glob("*.mp4"):
        file_name = video_path.stem  # Get the filename without extension
        print(f'Processing video: {file_name}')

        # Generate corresponding file paths
        input_landmarks_path = landmarks_folder / f"{file_name}_coordinates_local.parquet"
        annotations_path = annotations_folder / f"{file_name}_annotations.json"

        # Get video frame size
        capture = cv2.VideoCapture(str(video_path))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        # Get filename details
        competition, athlete_number, gender, ground_truth = extract_details(file_name)
        ground_truth_plus = "+" in str(ground_truth)

        # Detect holds separately for left and right hands
        left_hand_holds, right_hand_holds = process_video_and_track_holds(video_path, input_landmarks_path, annotations_path, control_threshold=60)

        # Detect fall
        fall_frame, fall_interval = detect_fall(input_landmarks_path)

        if fall_frame is not None:
            # Determine interval for plus progression
            check_start_frame, check_end_frame, last_controlled_hold, next_free_hold = get_interval_for_plus_progression(
                left_hand_holds, right_hand_holds, fall_frame, fall_interval
            )

            # Get the last controlled holds before the fall
            left_hand_hold, right_hand_hold = get_hand_positions_at_frame(
                input_landmarks_path=input_landmarks_path,
                annotations_path=annotations_path,
                check_start_frame=check_start_frame,
                width=width,
                height=height,
                last_controlled_hold=last_controlled_hold
            )

            # Check which hand moved more
            left_moved, right_moved = check_hand_movement_in_interval(
                input_landmarks_path, check_start_frame, check_end_frame, width, height
            )

            # Loop through different hand progression thresholds
            for hand_threshold in hand_threshold_values:
                print(f'Testing hand threshold: {hand_threshold}')

                # Calculate hand progression with the current threshold
                hand_progression = determine_hand_progression(
                    annotations_path, input_landmarks_path, next_free_hold,
                    left_hand_hold, right_hand_hold, left_moved, right_moved,
                    check_start_frame, check_end_frame, width, height,
                    hand_progression_threshold=hand_threshold  # Pass threshold here
                )

                # Store the results
                results.append({
                    "file_name": file_name,
                    "hand_progression_threshold": hand_threshold,
                    "hand_progression": hand_progression["hand_progression"],
                    "ground_truth_score": ground_truth,
                    "ground_truth_plus": ground_truth_plus
                })

    # Convert results to DataFrame
    plus_algorithm_results = pd.DataFrame(results)

    # Save the DataFrame as a .pd file
    plus_algorithm_results.to_pickle(output_dataframe_path)

    print(f"Results saved to {output_dataframe_path}")


def control_threshold_tuning():
    """Tunes the control threshold and evaluates accuracy."""

    # Set folder paths
    video_folder = Path("../data/input/videos/")
    landmarks_folder = Path("../data/input/landmarks/")
    annotations_folder = Path("../data/input/topos/")

    # Output file path
    output_dataframe_path = Path("../data/output/plus_algorithm/control_threshold_tuning.pd")

    # Set control thresholds to test (from 30 to 100 in steps of 10)
    control_threshold_values = list(range(30, 101, 10))

    # Function to extract details from the filename
    def extract_details(video_name):
        parts = video_name.split("_")

        competition = "lenzburg" if "lenzburg" in parts else "villars" if "villars" in parts else None
        gender = "male" if "men" in parts else "female" if "women" in parts else None

        athlete_number_match = re.search(r'n(\d+)', video_name)
        athlete_number = athlete_number_match.group(1) if athlete_number_match else None

        ground_truth_match = re.search(r'(\d+\+?|\d+)', parts[-1])
        ground_truth = ground_truth_match.group(1) if ground_truth_match else None

        return competition, athlete_number, gender, ground_truth

    # Initialize results list
    results = []

    # Iterate over all videos in the folder
    for video_path in video_folder.glob("*.mp4"):
        file_name = video_path.stem  # Get filename without extension
        print(f'Processing video: {file_name}')

        # Generate corresponding file paths
        input_landmarks_path = landmarks_folder / f"{file_name}_coordinates_local.parquet"
        annotations_path = annotations_folder / f"{file_name}_annotations.json"

        # Get video frame size
        capture = cv2.VideoCapture(str(video_path))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        # Get filename details
        competition, athlete_number, gender, ground_truth = extract_details(file_name)

        # Convert ground truth to numeric by removing the "+" sign
        ground_truth_control = int(re.sub(r'\+', '', ground_truth)) if ground_truth else None

        # Loop through different control thresholds
        for control_threshold in control_threshold_values:
            print(f'Testing control threshold: {control_threshold}')

            # Process holds with the current control threshold
            left_hand_holds, right_hand_holds = process_video_and_track_holds(video_path, input_landmarks_path,
                                                                              annotations_path, control_threshold)

            # Merge left and right hand holds
            all_holds = {**left_hand_holds, **right_hand_holds}

            # Get the last controlled hold
            last_controlled_hold = max(all_holds.keys()) if all_holds else None

            # Store the results
            results.append({
                "file_name": file_name,
                "control_threshold": control_threshold,
                "last_controlled_hold": last_controlled_hold,
                "ground_truth_score": ground_truth,
                "ground_truth_control": ground_truth_control
            })

    # Convert results to DataFrame
    control_algorithm_results = pd.DataFrame(results)

    # Save the DataFrame as a .pd file
    control_algorithm_results.to_pickle(output_dataframe_path)

    print(f"Results saved to {output_dataframe_path}")


def main():
    """Runs the full algorithm on all videos and saves results to a dataframe."""

    # Set thresholds
    control_threshold = 60
    hand_progression_threshold = 0.4
    hip_progression_threshold = 22

    # Set folder paths
    video_folder = Path("../data/input/videos/")
    landmarks_folder = Path("../data/input/landmarks/")
    annotations_folder = Path("../data/input/topos/")
    output_dataframe_path = Path("../data/output/plus_algorithm/final_algorithm_results.pd")

    # Function to extract details from the filename
    def extract_details(video_name):
        parts = video_name.split("_")

        competition = "lenzburg" if "lenzburg" in parts else "villars" if "villars" in parts else None
        gender = "male" if "men" in parts else "female" if "women" in parts else None

        athlete_number_match = re.search(r'n(\d+)', video_name)
        athlete_number = athlete_number_match.group(1) if athlete_number_match else None

        ground_truth_match = re.search(r'(\d+\+?|\d+)', parts[-1])
        ground_truth = ground_truth_match.group(1) if ground_truth_match else None
        ground_truth_control = int(ground_truth.replace("+", "")) if ground_truth else None
        ground_truth_plus = "+" in str(ground_truth)

        return competition, athlete_number, gender, ground_truth, ground_truth_control, ground_truth_plus

    # Initialize results list
    results = []

    # Iterate over all videos in the folder
    for video_path in video_folder.glob("*.mp4"):
        file_name = video_path.stem  # Get the filename without extension
        print(f'Processing video: {file_name}')

        # Generate corresponding file paths
        input_landmarks_path = landmarks_folder / f"{file_name}_coordinates_local.parquet"
        annotations_path = annotations_folder / f"{file_name}_annotations.json"

        # Get video frame size
        capture = cv2.VideoCapture(str(video_path))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        # Extract filename details
        competition, athlete_number, gender, ground_truth_score, ground_truth_control, ground_truth_plus = extract_details(file_name)

        # Detect holds separately for left and right hands
        left_hand_holds, right_hand_holds = process_video_and_track_holds(
            video_path, input_landmarks_path, annotations_path, control_threshold=control_threshold
        )

        # Detect fall
        fall_frame, fall_interval = detect_fall(input_landmarks_path)

        if fall_frame is not None:
            # Determine interval for plus progression
            check_start_frame, check_end_frame, last_controlled_hold, next_free_hold = get_interval_for_plus_progression(
                left_hand_holds, right_hand_holds, fall_frame, fall_interval
            )

            # Get the last controlled holds before the fall
            left_hand_hold, right_hand_hold = get_hand_positions_at_frame(
                input_landmarks_path=input_landmarks_path,
                annotations_path=annotations_path,
                check_start_frame=check_start_frame,
                width=width,
                height=height,
                last_controlled_hold=last_controlled_hold
            )

            # Check which hand moved more
            left_moved, right_moved = check_hand_movement_in_interval(
                input_landmarks_path, check_start_frame, check_end_frame, width, height
            )

            # Determine hand progression
            hand_progression = determine_hand_progression(
                annotations_path, input_landmarks_path, next_free_hold,
                left_hand_hold, right_hand_hold, left_moved, right_moved,
                check_start_frame, check_end_frame, width, height,
                hand_progression_threshold=hand_progression_threshold  # Pass threshold here
            )

            # Calculate hip progression
            hip_progression = calculate_hip_progression(
                input_landmarks_path, check_start_frame, check_end_frame, width, height,
                hip_threshold=hip_progression_threshold, interval=10
            )

            # Determine final plus decision
            is_plus = (
                hand_progression["hand_progression"] and
                (hip_progression["left_hip_progression"] or hip_progression["right_hip_progression"])
            )

            # Assign final score
            if is_plus:
                print("plus")
                print(f"Final score: {last_controlled_hold}+")
                final_score = f"{last_controlled_hold}+"
            else:
                print("no plus")
                print(f"Final score: {last_controlled_hold}")
                final_score = int(last_controlled_hold)

            # Store the results
            results.append({
                "file_name": file_name,
                "last_controlled_hold": last_controlled_hold,
                "hip_progression": hip_progression["left_hip_progression"] or hip_progression["right_hip_progression"],
                "hand_progression": hand_progression["hand_progression"],
                "ground_truth_score": ground_truth_score,
                "ground_truth_control": ground_truth_control,
                "ground_truth_plus": ground_truth_plus,
                "plus": is_plus,
                "final_score": final_score
            })

    # Convert results to DataFrame
    final_algorithm_results = pd.DataFrame(results)

    # Save the DataFrame as a .pd file
    final_algorithm_results.to_pickle(output_dataframe_path)

    print(f"Results saved to {output_dataframe_path}")


if __name__ == "__main__":
    main()

