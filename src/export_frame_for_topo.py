import cv2
from pathlib import Path

"""
This script allows you to select frames from a video for further processing.
It's used to export a frame to generate the digital topo.

Run code:
- Press the spacebar to select frames.
- Use the following keys to navigate frames:
  - 'a': Go back one frame. Stops at the first frame.
  - 'd': Skip forward one frame.
  - 'l': Jump directly to the last frame.
- Press 'q' to exit and save the chosen frames.

Other usecase: intrinsic calibration
For the intrinsic calibration, frames are chosen for each camera. 
This frame export can then also be uploaded to MATLAB's "camera calibrator" (20-80 frames).

Input:
- Climbing video / Camera calibration video

Output:
- Selected frames
"""

# Paths
file_name = "edited_villars_men_semifinals_n110_plus_14+"

# Automatically generated paths based on the file name
video_path = f'../data/input/videos/{file_name}.mp4'
output_path = f'../data/input/frames/{file_name}'

# Open video
cap = cv2.VideoCapture(str(video_path))
frames = []
frames_num = []

# Get the total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Current frame position (starts at 0)
current_frame = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# Interactive frame selection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display frame with annotation
    cv2.putText(
        frame, f"Number of selected frames: {len(frames)}", (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA
    )
    frame2 = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resized for display only
    cv2.imshow('Video', frame2)

    # Key controls
    key = cv2.waitKey(0) & 0xFF
    if key == 32:  # Spacebar: Select frame
        frames.append(frame)
        frames_num.append(current_frame)
    elif key == 97:  # 'a': Go back one frame
        if current_frame > 0:  # Prevent going below the first frame
            current_frame -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            print("Skipped back one frame.")
        else:
            print("Already at the first frame.")
    elif key == 100:  # 'd': Skip forward one frame
        if current_frame < total_frames - 1:  # Prevent exceeding the last frame
            current_frame += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            print("Skipped forward one frame.")
        else:
            print("Already at the last frame.")
    elif key == ord('l'):  # 'l': Jump to the last frame
        current_frame = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        print("Jumped to the last frame.")
    elif key == ord('q'):  # 'q': Quit
        break

cap.release()
cv2.destroyAllWindows()

# Save selected frames
print('Completed frame selection. Saving frames...')
for i, frame_num in enumerate(frames_num):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        output_file = f'{output_path}_frame{i}.jpg'
        print(f"Saving: {output_file}")
        cv2.imwrite(str(output_file), frame)
    cap.release()
cv2.destroyAllWindows()
