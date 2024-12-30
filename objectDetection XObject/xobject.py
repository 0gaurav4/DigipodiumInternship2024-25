import cv2
import os
from ultralytics import YOLO

def detect_objects_in_video(video_path, model):
    """Detect unique objects in the video."""
    detected_objects = set()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Process frames at intervals to speed up detection (e.g., every 10th frame)
    frame_interval = 10
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Only process every 10th frame to reduce computation
        if frame_count % frame_interval == 0:
            results = model.predict(frame, show=False)
            clss = results[0].boxes.cls.cpu().tolist()
            detected_objects.update(model.names[int(cls)] for cls in clss)

        frame_count += 1

    cap.release()
    return detected_objects

def save_combined_clips(video_path, model, output_dir, target_object="bird", show_preview=False):
    """Detect and save multiple clips around occurrences of the target object as a single combined video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS is zero, indicating an issue with the video file.")
        cap.release()
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    
    combined_output_path = os.path.join(output_dir, f"combined_{target_object}_video.mp4")
    combined_video_writer = cv2.VideoWriter(combined_output_path, codec, fps, (w, h))

    recording = False
    detection_times = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get current time in seconds
        results = model.predict(frame, show=False)
        clss = results[0].boxes.cls.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

        # Check if target object is detected in this frame
        detected = any(model.names[int(cls)] == target_object for cls in clss)

        # Start recording frames if target object is detected
        if detected:
            if not recording:
                recording = True
                start_time = current_time
                print(f"{target_object} detected! Adding frames starting at {start_time:.2f} seconds.")
            # Highlight and write frames to the combined video
            for i, cls in enumerate(clss):
                if model.names[int(cls)] == target_object:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Highlight with green box
                    cv2.putText(frame, target_object, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            combined_video_writer.write(frame)
            if show_preview:
                cv2.imshow("Combined Clip Preview", frame)

        # Stop recording if the target object is no longer detected
        if recording and not detected:
            recording = False
            end_time = current_time
            detection_times.append((start_time, end_time))
            print(f"Stopped adding frames at {end_time:.2f} seconds.")

        if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    combined_video_writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"Combined video saved as '{combined_output_path}'.")
    print("Detection times for each occurrence:")
    for i, (start, end) in enumerate(detection_times, 1):
        print(f"Occurrence #{i}: Start = {start:.2f} sec, End = {end:.2f} sec")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    input_video_path = r"C:\Users\gaura\Desktop\objectDetection\uploads\car.mp4"
    output_directory = r"C:\Users\gaura\Desktop\objectDetection\saved_videos"

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Step 1: Detect objects in the video (only the actual objects detected)
    detected_objects = detect_objects_in_video(input_video_path, model)

    if detected_objects:
        print("Detected objects in the video:", detected_objects)
        target_object = input("Enter the name of the object you want to track: ").strip()
        
        if target_object in detected_objects:
            save_combined_clips(input_video_path, model, output_directory, target_object, show_preview=True)
        else:
            print(f"{target_object} not found in the video.")
    else:
        print("No objects detected in the video.")
