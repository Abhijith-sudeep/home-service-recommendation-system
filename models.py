from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

db = SQLAlchemy()

# -------------------- USER MODEL --------------------

class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(100), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    # 📍 NEW: User Location
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    # Relationship
    jobs = db.relationship("Job", backref="user", lazy=True)


# -------------------- WORKER MODEL --------------------

class Worker(db.Model):
    __tablename__ = "worker"

    id = db.Column(db.Integer, primary_key=True)

    first_name = db.Column(db.String(100), nullable=False)

    last_name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(100), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    service_type = db.Column(db.String(100), nullable=False)
    
    description = db.Column(db.String(300))

    # 📍 Location (already exists but keep it clean)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    works_done = db.Column(db.Integer, default=0)

    # Relationship
    reviews = db.relationship("Review", backref="worker", lazy=True)


# -------------------- JOB / BOOKING MODEL --------------------

class Job(db.Model):
    __tablename__ = "job"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    worker_id = db.Column(db.Integer, db.ForeignKey("worker.id"), nullable=False)

    status = db.Column(db.String(50), default="Hired")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# -------------------- REVIEW MODEL --------------------

class Review(db.Model):
    __tablename__ = "review"

    id = db.Column(db.Integer, primary_key=True)

    worker_id = db.Column(db.Integer, db.ForeignKey("worker.id"), nullable=False)

    # ⭐ MULTI-RATING SYSTEM
    quality = db.Column(db.Integer)
    punctuality = db.Column(db.Integer)
    professionalism = db.Column(db.Integer)
    reliability = db.Column(db.Integer)

    # ⭐ Optional overall rating (derived)
    rating = db.Column(db.Float)

    comment = db.Column(db.String(300))


# -------------------- MESSAGE MODEL --------------------

class Message(db.Model):

    __tablename__ = "message"

    id = db.Column(db.Integer, primary_key=True)

    sender_id = db.Column(db.Integer)

    receiver_id = db.Column(db.Integer)

    sender_type = db.Column(db.String(10))

    content = db.Column(db.String(500))

    timestamp = db.Column(
        db.DateTime,
        default=lambda: datetime.utcnow() + timedelta(hours=5, minutes=30)
    )

    seen = db.Column(db.Boolean, default=False)