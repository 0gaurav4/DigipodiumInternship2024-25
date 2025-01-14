from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Video(db.Model):
    """Stores metadata about uploaded videos."""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False)
    output_dir = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<Video {self.filename}>"


class Clip(db.Model):
    """Stores details of generated clips."""
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    clip_filename = db.Column(db.String(255), nullable=False)
    clip_duration = db.Column(db.Integer, nullable=False)
    video = db.relationship("Video", backref=db.backref("clips", cascade="all, delete-orphan", lazy=True))

    def __repr__(self):
        return f"<Clip {self.clip_filename}>"
