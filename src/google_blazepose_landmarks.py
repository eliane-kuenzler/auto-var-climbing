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

"""
Generating gloabl and local coordinates per frame of each climbing video. 
Using mediapipe's PoseLandmarker to detect 33 keypoints on the human body.

Input: 
- mp4 videos

Output: 
- global coordinates
- local coordinates
"""

def save_df_as_parquet(input_df: pd.DataFrame, output_path: Path):
    table = pyarrow.Table.from_pandas(df=input_df)
    parquet.write_table(table, str(output_path))


def pose_estimation(input_video_path: Path, output_local_path: Path, output_global_path: Path):
    print(f"Processing video: {input_video_path.name}")
    process_capture = cv2.VideoCapture(str(input_video_path))
    if not process_capture.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    detector = vision.PoseLandmarker.create_from_options(options)
    print("Pose detector initialized successfully.")

    columns = ["time(s)"]
    for key in keypoint_dict.keys():
        columns.append(key + "_x")
        columns.append(key + "_y")
        columns.append(key + "_z")
        columns.append(key + "_v")
        columns.append(key + "_p")

    frame_rate = process_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(process_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frame rate: {frame_rate}, Total frames: {total_frames}")

    flat_list_world_df = pd.DataFrame(columns=columns).astype("float32")
    flat_list_pose_df = pd.DataFrame(columns=columns).astype("float32")

    for current_frame_num in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = process_capture.read()
        if not ret:
            break

        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_RGB)
        timestamp = mp.Timestamp.from_seconds(
            process_capture.get(cv2.CAP_PROP_POS_FRAMES) / frame_rate
        )
        results = detector.detect_for_video(mp_image, int(timestamp.seconds() * 1000))
        results_pose_landmarks = results.pose_landmarks
        results_world_landmarks = results.pose_world_landmarks

        if results_pose_landmarks and len(results_pose_landmarks) == 1:
            flat_list_pose = [
                coordinate for landmark in results_pose_landmarks[0]
                for coordinate in [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
            ]
        else:
            flat_list_pose = [np.NaN] * (33 * 5)

        if results_world_landmarks and len(results_world_landmarks) == 1:
            flat_list_world = [
                coordinate for landmark in results_world_landmarks[0]
                for coordinate in [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
            ]
        else:
            flat_list_world = [np.NaN] * (33 * 5)

        flat_list_pose.insert(0, (current_frame_num - 1) / frame_rate)
        flat_list_world.insert(0, (current_frame_num - 1) / frame_rate)

        flat_list_world_df.loc[len(flat_list_world_df)] = flat_list_world
        flat_list_pose_df.loc[len(flat_list_pose_df)] = flat_list_pose

    print("\nFinished processing video.")
    save_df_as_parquet(flat_list_pose_df, output_local_path)
    save_df_as_parquet(flat_list_world_df, output_global_path)
    process_capture.release()


if __name__ == "__main__":
    video_path = Path('../data/input/videos')
    output_coordinates_path = Path('../data/input/landmarks')
    model_path = Path('../models/pose_landmarker_full.task')

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = BaseOptions(model_asset_path=model_path)

    options = PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VisionRunningMode.VIDEO,
    )

    keypoint_dict = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye_center": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye_center": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "left_mouth": 9,
        "right_mouth": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot": 31,
        "right_foot": 32,
    }

    # Process only a specific video
    specific_video_file = video_path / "edited_villars_men_semifinals_n110_plus_14+"
    output_global_coordinates_path = output_coordinates_path / f"{specific_video_file.stem}_coordinates_global.parquet"
    output_local_coordinates_path = output_coordinates_path / f"{specific_video_file.stem}_coordinates_local.parquet"

    pose_estimation(specific_video_file, output_local_coordinates_path, output_global_coordinates_path)

    """
    # Commented out code for processing all videos in the folder
    for video_file in video_path.glob("*.mp4"): # check if .avi or .mp4
        output_global_coordinates_path = output_coordinates_path / f"{video_file.stem}_coordinates_global.parquet"
        output_local_coordinates_path = output_coordinates_path / f"{video_file.stem}_coordinates_local.parquet"
        pose_estimation(video_file, output_local_coordinates_path, output_global_coordinates_path)
    """