import cv2
from rtmlib import Wholebody, draw_skeleton
from tqdm import tqdm

"""
This script demonstrates how to use the Wholebody model for real-time multi-person pose estimation.

Link to github repo:
https://github.com/Tau-J/rtmlib/tree/main

Input:
- Climbing video

Output:
- Video with rmto keypoints overlaid
"""

# File paths
file_name = "n43_noplus_15"

# Automatically generated paths based on the file name
input_video_path = f'../data/input/videos/{file_name}.mp4'
output_video_path = f'../data/output/{file_name}_rtmo.mp4'

device = 'cpu'  # or 'cuda' if GPU is available
backend = 'onnxruntime'
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# Initialize the Wholebody model
wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='performance',  # 'balanced', 'performance', 'lightweight'
                      backend=backend, device=device)

cap = cv2.VideoCapture(input_video_path)

# video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize the progress bar
with tqdm(total=total_frames, desc="Processing video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Detect skeleton keypoints and scores
        keypoints, scores = wholebody(frame)

        # Draw skeleton on the frame
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        # Write the processed frame to the output video
        out.write(img_show)

        # Update the progress bar
        pbar.update(1)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")