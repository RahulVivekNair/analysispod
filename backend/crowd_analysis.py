import os
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import json
import matplotlib.pyplot as plt
def crowd_analysis(file_path, output_path, settings):
    output_filename = os.path.join(output_path, "crowd_analysis_output.mp4")
     # Load the video from the temporary folder
    video = cv2.VideoCapture(file_path)
    # Load the YOLOv5 model

    model = YOLO("models/yolov9e.pt",task='detect')# Use the GPU for inference
    model.to("cuda")
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=0.5,
        radius=25,
        kernel_size=25,
        top_hue=0,
        low_hue=125,
    )
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    corner_annotator = sv.BoxCornerAnnotator()
    trace_annotator = sv.TraceAnnotator()
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    byte_tracker = sv.ByteTrack(
        track_thresh=0.35,
        track_buffer=5 * fps,
        match_thresh=0.99,
        frame_rate=fps,
    )
    video_info = sv.VideoInfo.from_video_path(video_path=file_path )
    frames_generator = sv.get_video_frames_generator(
        source_path=file_path , stride=1
    )
    crowd_counts = []  # Store crowd count for each frame
    timestamps = []
    with sv.VideoSink(target_path=output_filename, video_info=video_info,codec="H264") as sink:
        frame_count = 0
        for frame in frames_generator:
            result = model(
                source=frame,
                classes=[0],  # only person class
                conf=settings["confidence_threshold"],
                iou=0.5,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=0,  # use None = CPU, 0 = single GPU, or [0,1] = dual GPU
            )[0]

            detections = sv.Detections.from_ultralytics(result)  # get detections

            detections = byte_tracker.update_with_detections(
                detections
            )  # update tracker

            annotated_frame = frame.copy()  # Start with a fresh copy in every iteration

            if settings["crowd_heatmap"]:
                annotated_frame = heat_map_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

            if settings["crowd_trace"]:
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

            if settings["crowd_bounding_box"]:
                annotated_frame = corner_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
            if settings["crowd_label"]:
            ### draw other attributes from `detections` object
                labels = [
                    f"#{tracker_id}"
                    for class_id, tracker_id in zip(
                        detections.class_id, detections.tracker_id
                    )
                ]
                
                label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            crowd_count = len(detections)  
            cv2.putText(annotated_frame, f"Crowd Count: {crowd_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            crowd_counts.append(crowd_count)
            timestamps.append(frame_count / fps)
            sink.write_frame(frame=annotated_frame)
            frame_count += 1
    # Perform crowd analysis
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed 
    plt.plot(timestamps, crowd_counts)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of People")
    plt.title("Crowd Analysis Graph")
    plt.grid(True) 
    plt.tight_layout()  # Improve layout
    plt.savefig(os.path.join(output_path, "crowd_analysis_graph.png"))
    # Return the results
    return {"message": "Crowd analysis complete"}