from pathlib import Path
import pandas as pd
import pyarrow
from pyarrow import parquet
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import time

"""
This script generates global and local coordinates from MediaPipe.
Enables to compare the accuracy of the MediaPipe pose estimation model.

Input: 
- .avi and .mp4 videos

Output: 
- global coordinates
- local coordinates
- Pose detection percentage
- FPS (frames per second)
"""


def save_df_as_parquet(input_df: pd.DataFrame, output_path: Path):
    table = pyarrow.Table.from_pandas(df=input_df)
    parquet.write_table(table, str(output_path))


def pose_estimation(input_video_path: Path, output_local_path: Path, output_global_path: Path, detector):
    print(f"Processing video: {input_video_path.name}")
    process_capture = cv2.VideoCapture(str(input_video_path))
    if not process_capture.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return None, None  # Return None if video can't be opened

    frame_rate = process_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(process_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frame rate: {frame_rate}, Total frames: {total_frames}")

    # DataFrames for pose and world landmark data
    columns = ["time(s)"]
    for key in keypoint_dict.keys():
        columns.append(f"{key}_x")
        columns.append(f"{key}_y")
        columns.append(f"{key}_z")
        columns.append(f"{key}_v")
        columns.append(f"{key}_p")

    flat_list_world_df = pd.DataFrame(columns=columns).astype("float32")
    flat_list_pose_df = pd.DataFrame(columns=columns).astype("float32")

    pose_detected_frames = 0  # Count of frames where pose is detected
    total_frames_processed = 0  # Total frames processed

    start_time = time.time()  # Start the timer for FPS

    # Reset video timestamp for each video
    timestamp = 0  # Reset timestamp to 0 at the start of each video

    for current_frame_num in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = process_capture.read()
        if not ret:
            break

        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_RGB)

        # Calculate frame time explicitly based on the frame index and frame rate
        frame_time = current_frame_num / frame_rate  # Time in seconds
        mp_timestamp = mp.Timestamp.from_seconds(frame_time)  # Use frame time directly

        # Detect pose landmarks
        results = detector.detect_for_video(mp_image, int(mp_timestamp.seconds() * 1000))

        results_pose_landmarks = results.pose_landmarks
        results_world_landmarks = results.pose_world_landmarks

        # Check if pose landmarks are detected
        if results_pose_landmarks:
            flat_list_pose = [
                coordinate for landmark in results_pose_landmarks[0]
                for coordinate in [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
            ]
            pose_detected_frames += 1  # Increment count for detected poses
        else:
            flat_list_pose = [np.NaN] * (33 * 5)

        # Process world landmarks
        if results_world_landmarks:
            flat_list_world = [
                coordinate for landmark in results_world_landmarks[0]
                for coordinate in [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
            ]
        else:
            flat_list_world = [np.NaN] * (33 * 5)

        # Insert timestamp and append to DataFrame
        flat_list_pose.insert(0, frame_time)
        flat_list_world.insert(0, frame_time)

        flat_list_world_df.loc[len(flat_list_world_df)] = flat_list_world
        flat_list_pose_df.loc[len(flat_list_pose_df)] = flat_list_pose
        total_frames_processed += 1

    # Calculate FPS (frames processed per second)
    end_time = time.time()
    processing_time = end_time - start_time
    fps = total_frames_processed / processing_time  # FPS: frames per second

    # Calculate pose detection percentage
    pose_detection_percentage = (pose_detected_frames / total_frames_processed) * 100

    print(f"\nFinished processing video: {input_video_path.name}")
    print(f"Pose detection percentage: {pose_detection_percentage}%")
    print(f"FPS: {fps}")

    # Save results as parquet files
    save_df_as_parquet(flat_list_pose_df, output_local_path)
    save_df_as_parquet(flat_list_world_df, output_global_path)
    process_capture.release()

    return pose_detection_percentage, fps


if __name__ == "__main__":
    # set path to process video
    video_path = Path('../../data/input')
    output_coordinates_path = Path('../../data/output')
    model_path = Path('../../models/pose_landmarker_full.task')

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VisionRunningMode.VIDEO,
    )

    keypoint_dict = {
        "nose": 0, "left_eye_inner": 1, "left_eye_center": 2, "left_eye_outer": 3,
        "right_eye_inner": 4, "right_eye_center": 5, "right_eye_outer": 6, "left_ear": 7,
        "right_ear": 8, "left_mouth": 9, "right_mouth": 10, "left_shoulder": 11,
        "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15,
        "right_wrist": 16, "left_pinky": 17, "right_pinky": 18, "left_index": 19,
        "right_index": 20, "left_thumb": 21, "right_thumb": 22, "left_hip": 23,
        "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27,
        "right_ankle": 28, "left_heel": 29, "right_heel": 30, "left_foot": 31,
        "right_foot": 32,
    }

    # Initialize detector
    detector = vision.PoseLandmarker.create_from_options(options)
    print("Pose detector initialized successfully.")

    # Store metrics for each video
    all_metrics = []

    # Loop through both .avi and .mp4 files
    video_files = list(video_path.glob("*.[mM][pP]4")) + list(video_path.glob("*.[aA][vV][iI]"))
    print("Found video files:", video_files)

    for video_file in video_files:
        output_global_coordinates_path = output_coordinates_path / f"{video_file.stem}_coordinates_global.parquet"
        output_local_coordinates_path = output_coordinates_path / f"{video_file.stem}_coordinates_local.parquet"

        # Process each video and get metrics
        pose_detection_percentage, fps = pose_estimation(video_file, output_local_coordinates_path,
                                                         output_global_coordinates_path, detector)

        if pose_detection_percentage is not None and fps is not None:
            # Store the metrics for this video
            all_metrics.append({
                "video": video_file.name,
                "pose_detection_percentage": pose_detection_percentage,
                "fps": fps,
            })

    # Calculate the averages
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate and print the averages
        average_pose_detection_percentage = metrics_df["pose_detection_percentage"].mean()
        average_fps = metrics_df["fps"].mean()

        print("\nAverage Pose Detection Percentage:", average_pose_detection_percentage)
        print("Average FPS:", average_fps)

        # Optionally, save the metrics to a CSV or Parquet file
        # metrics_df.to_csv("pose_estimation_metrics.csv", index=False)
    else:
        print("No metrics were collected. Check if the videos were processed correctly.")
