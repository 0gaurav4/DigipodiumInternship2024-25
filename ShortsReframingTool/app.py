from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from models import db, Video, Clip
from datetime import datetime
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'output_clips/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shorts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize database
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/Reframe', methods=['GET', 'POST'])
def Reframe():
    return render_template('Reframe.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

def reframe_video_to_shorts_in_clips(video_path, output_dir, model_path="yolov8n.pt", shorts_size=(1080, 1920), clip_duration=60):
    """Reframe video to 9:16 aspect ratio and save 1-minute clips with a preview of detected persons."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False

    model = YOLO(model_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    clip_frame_count = clip_duration * fps

    shorts_width, shorts_height = shorts_size
    original_aspect = original_width / original_height
    shorts_aspect = shorts_width / shorts_height

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

        # Crop and resize for 9:16 aspect ratio
        if original_aspect > shorts_aspect:
            new_width = int(original_height * shorts_aspect)
            x_offset = (original_width - new_width) // 2
            cropped_frame = frame[:, x_offset:x_offset + new_width]
        else:
            new_height = int(original_width / shorts_aspect)
            y_offset = (original_height - new_height) // 2
            cropped_frame = frame[y_offset:y_offset + new_height, :]

        resized_frame = cv2.resize(cropped_frame, (shorts_width, shorts_height))
        out.write(resized_frame)

        frame_count += 1
        if frame_count >= clip_frame_count:
            out.release()
            current_clip_index += 1
            out, current_output_path = get_output_writer(current_clip_index)
            frame_count = 0

    if frame_count > 0 and frame_count < clip_frame_count:
        out.release()

    cap.release()
    return True




@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files or not request.files['video'].filename:
            return jsonify({'error': 'No video uploaded'}), 400

        video = request.files['video']
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        # Save video metadata in the database
        new_video = Video(filename=filename, upload_time=datetime.utcnow(), output_dir="")
        db.session.add(new_video)
        db.session.commit()

        # Start video reframing process
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], str(new_video.id))
        os.makedirs(output_dir, exist_ok=True)
        success = reframe_video_to_shorts_in_clips(video_path, output_dir)
        if not success:
            return jsonify({'error': 'Failed to process video'}), 500

        # Save generated clips to database
        for clip_name in os.listdir(output_dir):
            new_clip = Clip(video_id=new_video.id, clip_filename=clip_name, clip_duration=60)
            db.session.add(new_clip)
        db.session.commit()

        return redirect(url_for('result'))

    return render_template('upload.html')



@app.route('/result')
def result():
    videos = Video.query.all()
    return render_template('result.html', videos=videos)


@app.route('/reframe/<int:video_id>')
def reframe_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404

    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], str(video.id))
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)

    success = reframe_video_to_shorts_in_clips(video_path, output_dir)
    if not success:
        return jsonify({'error': 'Failed to process video'}), 500

    # Save generated clips to database
    for clip_name in os.listdir(output_dir):
        new_clip = Clip(video_id=video.id, clip_filename=clip_name, clip_duration=60)
        db.session.add(new_clip)
    db.session.commit()

    return redirect(url_for('result'))


@app.route('/download/<int:video_id>/<clip_filename>')
def download_clip(video_id, clip_filename):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404

    clip_path = os.path.join(video.output_dir, clip_filename)
    if not os.path.exists(clip_path):
        return jsonify({'error': 'Clip not found'}), 404

    return send_from_directory(video.output_dir, clip_filename, as_attachment=True)


@app.route('/delete/<int:video_id>')
def delete_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return redirect(url_for('result'))

    for clip in video.clips:
        os.remove(os.path.join(video.output_dir, clip.clip_filename))
    os.rmdir(video.output_dir)
    db.session.delete(video)
    db.session.commit()

    return redirect(url_for('result'))

if __name__ == '__main__':
    app.run(debug=True)
