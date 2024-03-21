import os
import cv2
from ultralytics import YOLO
import numpy as np

def fire_analysis(file_path, output_path):
    output_filename = os.path.join(output_path, "fire_and_smoke_analysis_output.mp4")
    video = cv2.VideoCapture(file_path)
    model = YOLO("models/fire_yolov8.pt")  # Load the YOLOv8 model
    model.to("cuda")  # Use the GPU for inference
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = video.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame, verbose=False)  # Run the YOLOv8 model on the frame

        annotated_frame = frame.copy()  # Start with a fresh copy of the frame

        for res in results:
            annotated_frame = res.plot()  # Draw bounding boxes and labels on the frame

        out.write(annotated_frame)  # Write the annotated frame to the output video

    video.release()
    out.release()