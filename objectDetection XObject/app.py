from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
import os
import cv2
from ultralytics import YOLO
from models import db, Video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SAVED_VIDEOS_FOLDER'] = 'saved_videos/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'  # Use SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()  # Create database tables

class UploadForm(FlaskForm):
    video_file = FileField('Video File', validators=[DataRequired()])
    submit = SubmitField('Upload')

@app.route('/')
def index():
    form = UploadForm()
    return render_template('upload.html', form=form)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    list = os.listdir(app.config['SAVED_VIDEOS_FOLDER'] or os.getcwd())
    get = request.args.get('filename') and request.args.get('filepath')
    if get:
        list = [x for x in list if get in x]
    list = sorted(list)
    print(list)
    return render_template('result.html', list=list)

#logic to delete items in saved_videos folder
@app.route('/delete/<item>')
def delete_file(item):
    path = os.path.join(app.config['SAVED_VIDEOS_FOLDER'], item)
    if os.path.isfile(path):
        os.remove(path)
    return redirect('/result')

@app.route('/download/<item>')
def download_file(item):
    file_path = os.path.join(app.config['SAVED_VIDEOS_FOLDER'], item)
    if os.path.isfile(file_path):
        return send_from_directory(app.config['SAVED_VIDEOS_FOLDER'], item, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404
    

@app.route('/uploaded_videos')
def uploaded_videos():
    videos = Video.query.all()
    return render_template('uploaded_videos.html', videos=videos)

@app.route('/delete_video/<int:id>')
def delete_video(id):
    video = Video.query.get(id)
    db.session.delete(video)
    db.session.commit()
    return redirect('/uploaded_videos')

@app.route('/download_video/<int:id>')
def download_video(id):
    video = Video.query.get(id)
    return send_from_directory(app.config['UPLOAD_FOLDER'], video.filename, as_attachment=True)

@app.route('/', methods=['POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        video_file = form.video_file.data
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        # Run the detection
        model = YOLO("yolov8n.pt")
        detected_objects = detect_objects_in_video(video_path, model)

        if detected_objects is not None:
            # Save video info to the database
            new_video = Video(filename=video_file.filename, detected_objects=', '.join(detected_objects))
            db.session.add(new_video)
            db.session.commit()

            return jsonify({'detected_objects': list(detected_objects)})
        else:
            return jsonify({'error': 'Failed to process video'}), 500

    return render_template('upload.html', form=form)

@app.route('/track', methods=['POST'])
def track():
    target_object = request.form['target_object']
    video_path = request.form['video_path']

    # Run the tracking and save combined clips
    model = YOLO("yolov8n.pt")
    save_combined_clips(video_path, model, app.config['SAVED_VIDEOS_FOLDER'], target_object)

    return jsonify({'message': f'Tracking {target_object} completed! Combined clips saved view the result page.'})

def detect_objects_in_video(video_path, model):
    detected_objects = set()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_interval = 10
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            results = model.predict(frame, show=False)
            clss = results[0].boxes.cls.cpu().tolist()
            detected_objects.update(model.names[int(cls)] for cls in clss)

        frame_count += 1

    cap.release()
    return detected_objects

def save_combined_clips(video_path, model, output_dir, target_object):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    combined_output_path = os.path.join(output_dir, f"combined_{target_object}_video.mp4")
    combined_video_writer = cv2.VideoWriter(combined_output_path, codec, fps, (w, h))

    recording = False
    detection_times = []
    show_preview = True  # Set to True if you want to see the preview

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, show=False)
        clss = results[0].boxes.cls.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

        detected = any(model.names[int(cls)] == target_object for cls in clss)

        if detected:
            if not recording:
                recording = True
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get start time in seconds
            for i, cls in enumerate(clss):
                if model.names[int(cls)] == target_object:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Highlight with green box
                    cv2.putText(frame, target_object, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            combined_video_writer.write(frame)

        if recording and not detected:
            recording = False
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get current time in seconds
            detection_times.append((start_time, end_time))
            print(f"Stopped adding frames at {end_time:.2f} seconds.")
            jsonify({'detection_times': detection_times})
            

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
    jsonify({'detection_times': detection_times})

    return combined_output_path ,jsonify({'detection_times': detection_times})

if __name__ == "__main__":
    app.run(debug=True)