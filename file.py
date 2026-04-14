import cv2
import os
import numpy as np 


VIDEO_PATH = 'Disruptive Dan.mp4' 
OUTPUT_DIR = 'Disruptive_Dan_Frames_20' 
NUM_FRAMES_TO_SAVE = 20                 


os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}. Make sure it's in the same directory as the script.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames in video: {total_frames}")
print(f"Frames per second (FPS): {fps}")


if total_frames < NUM_FRAMES_TO_SAVE:
    print(f"Warning: Video has only {total_frames} frames. Will save all of them.")
    frame_indices = list(range(total_frames))
else:
    
    frame_indices_float = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_SAVE)
    
    frame_indices = sorted(list(set(np.round(frame_indices_float).astype(int))))

print(f"Frames to be extracted (indices): {frame_indices}")


saved_count = 0
for target_frame_number in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
    
    success, frame = cap.read()

    if success:
        timestamp_seconds = target_frame_number / fps
        
        frame_filename = os.path.join(
            OUTPUT_DIR, 
            f"frame_{saved_count+1:02d}_at_index_{target_frame_number:04d}_time_{timestamp_seconds:.1f}s.jpg"
        )
        
        cv2.imwrite(frame_filename, frame)
        print(f" Saved frame {saved_count+1}/{len(frame_indices)}: {frame_filename}")
        
        saved_count += 1


cap.release()
print("\n--- Finished extracting 20 frames ---")