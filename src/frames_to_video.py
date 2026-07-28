import cv2
import os
import glob
from pathlib import Path

def convert_frames_to_mp4(trial_folder, fps=30.0):
    # Find all frame images in the folder (e.g., frame1.jpg, frame2.jpg, ...)
    # Using sorting to ensure frames are stitched in the correct chronological order
    image_files = sorted(glob.glob(os.path.join(trial_folder, "frame*.jpg")), 
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    if not image_files:
        print(f"No frames found in {trial_folder}!")
        return

    # Read the first frame to get dimensions (width, height)
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape
    
    output_video_path = os.path.join(trial_folder, "output_compiled.mp4")
    
    # Use 'mp4v' or 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Compiling {len(image_files)} frames into {output_video_path}...")
    
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
            
    out.release()
    print("Video compilation complete and saved successfully!")

if __name__ == "__main__":
    # Example usage: point this directly to your trial folder path
    # e.g., 'Sensor/Trial_20260728_200335'
    target_folder = input("Enter the path to the trial folder (e.g., Sensor/Trial_...): ").strip()
    if os.path.exists(target_folder):
        convert_frames_to_mp4(target_folder)
    else:
        print("Error: The specified folder does not exist.")