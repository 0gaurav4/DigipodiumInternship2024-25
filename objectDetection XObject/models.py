from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    detected_objects = db.Column(db.Text, nullable=True)  # Store detected objects as a comma-separated strin
    

    def __repr__(self):
        return f'<Video {self.filename}>'
    
