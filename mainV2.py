import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process a football match video.")
parser.add_argument("--input_video_path", type=str, default="footballmatch.mp4", help="Path to the input video file.")
parser.add_argument("--output_video_path", type=str, default="output_video.mp4", help="Path to save the output video file.")
parser.add_argument("--model_path", type=str, default="./models/best.pt", help="Path to the model file.")
parser.add_argument("--track_stub", type=str, default="stubs/track_stubs.pkl", help="Path to track stub file.")
parser.add_argument("--camera_movement_stub", type=str, default="stubs/camera_movement_stub.pkl", help="Path to camera movement stub file.")
parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process at one time.")

args = parser.parse_args()

# Define paths
input_video_path = Path(__file__).parent / "input_videos" / args.input_video_path
output_video_path = Path(__file__).parent / "output_videos" / args.output_video_path
model_path = Path(__file__).parent / args.model_path

cap = cv2.VideoCapture(str(input_video_path))

# Get the FPS from the input video
input_fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer with the same FPS as the input video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video_path), fourcc, input_fps, (int(cap.get(3)), int(cap.get(4))))

# Load YOLO model
model = YOLO(model_path)

# Process video frames
MAX_FRAMES = args.max_frames
frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    frame_count += 1

    # Process frames in batches
    if len(frames) >= MAX_FRAMES or not ret:
        results = model.track(frames, persist=True)  # Track objects in the frames

        # Draw rings around detected players
        for i, result in enumerate(results):
            for box in result.boxes:
                if box.conf > 0.5:  # Filter out low-confidence detections
                    x1, y1, x2, y2 = box.xyxy[0]  # Accessing the first element of the tensor
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.circle(frames[i], center, radius, (0, 255, 0), 2)  # Draw a green ring
            out.write(frames[i])
        frames = []  # Reset frames list after processing

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

