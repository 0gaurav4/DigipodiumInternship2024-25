import cv2
import numpy as np
import os
from ultralytics import YOLO

# video drive link
# https://drive.google.com/uc?id=1v11kPvWuVgUh3UF7OdbZ_f2oImtHOzvS&export=download

def reframe_video_to_shorts_in_clips(video_path, output_dir, model_path="yolov8n.pt", shorts_size=(1080, 1920), clip_duration=60):
    """Reframe video to 9:16 aspect ratio and save 1-minute clips with a preview of detected persons."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Load YOLO model
    model = YOLO(model_path)

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frame_count = clip_duration * fps

    shorts_width, shorts_height = shorts_size
    original_aspect = original_width / original_height
    shorts_aspect = shorts_width / shorts_height

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    current_clip_index = 1
    frame_count = 0

    def get_output_writer(index):
        output_path = os.path.join(output_dir, f"shorts_clip_{index:02d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, shorts_size), output_path

    out, current_output_path = get_output_writer(current_clip_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        results = model.predict(frame, show=False)
        detections = results[0].boxes
        clss = detections.cls.cpu().numpy()
        boxes = detections.xyxy.cpu().numpy()

        # Filter detections for 'person' class
        person_class_id = 0  # YOLO's 'person' class ID
        person_boxes = [box for i, box in enumerate(boxes) if int(clss[i]) == person_class_id]

        # Determine how to reframe for portrait aspect
        if original_aspect > shorts_aspect:
            # Crop horizontally to center the frame
            new_width = int(original_height * shorts_aspect)
            x_offset = (original_width - new_width) // 2
            cropped_frame = frame[:, x_offset:x_offset + new_width]
            resized_frame = cv2.resize(cropped_frame, (shorts_width, shorts_height))
        else:
            # Crop vertically to center the frame
            new_height = int(original_width / shorts_aspect)
            y_offset = (original_height - new_height) // 2
            cropped_frame = frame[y_offset:y_offset + new_height, :]
            resized_frame = cv2.resize(cropped_frame, (shorts_width, shorts_height))

        # Write the reframed frame to the current clip
        out.write(resized_frame)

        # Highlight persons in the preview frame
        preview_frame = resized_frame.copy()
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1 - x_offset) if original_aspect > shorts_aspect else x1
            x2 = max(0, x2 - x_offset) if original_aspect > shorts_aspect else x2
            y1 = max(0, y1 - y_offset) if original_aspect <= shorts_aspect else y1
            y2 = max(0, y2 - y_offset) if original_aspect <= shorts_aspect else y2

            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display preview frame
        cv2.imshow('Shorts Frame (Preview)', preview_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Handle clip splitting
        frame_count += 1
        if frame_count >= clip_frame_count:
            out.release()
            print(f"Clip saved at: {current_output_path}")
            current_clip_index += 1
            out, current_output_path = get_output_writer(current_clip_index)
            frame_count = 0

    # Save any remaining frames
    if frame_count > 0:
        out.release()
        print(f"Final clip saved at: {current_output_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"All clips saved in {output_dir}")

if __name__ == "__main__":
    video_input_path = r"C:\Users\gaura\Downloads\python practice\pp\video.mp4"
    output_directory = r"C:\Users\gaura\Downloads\python practice\pp\shorts_clips"

    # Reframe video to shorts in clips
    reframe_video_to_shorts_in_clips(video_input_path, output_directory)
