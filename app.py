from flask import Flask, flash, render_template, request, redirect, session, url_for
from models import db, User, Worker, Review, Job, Message
from ml.ml_model import check_toxic
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
import math
from sklearn.preprocessing import MinMaxScaler


def calculate_distance(u_lat, u_lon, w_lat, w_lon):
    if not u_lat or not u_lon or not w_lat or not w_lon:
        return 9999  # far away if missing

    # Convert to radians
    u_lat, u_lon, w_lat, w_lon = map(math.radians, [u_lat, u_lon, w_lat, w_lon])

    dlat = w_lat - u_lat
    dlon = w_lon - u_lon

    a = math.sin(dlat/2)**2 + math.cos(u_lat) * math.cos(w_lat) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371  # Earth radius in KM

    return round(R * c, 2)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()


# -------------------- HELPER: CENSOR TEXT --------------------

def censor_text(text):
    words = text.split()
    censored_words = []

    for word in words:
        if len(word) > 2:
            censored = word[0] + "*"*(len(word)-2) + word[-1]
            censored_words.append(censored)
        else:
            censored_words.append(word)

    return " ".join(censored_words)


# -------------------- HOME --------------------

@app.route("/")
def home():
    return render_template("home.html")


# -------------------- USER REGISTRATION --------------------

@app.route("/register_user", methods=["GET", "POST"])
def register_user():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        # 🔍 VALIDATION
        if not name or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for("register_user"))

        # 📍 LOCATION CHECK
        if not latitude or not longitude:
            flash("Please select location!", "danger")
            return redirect(url_for("register_user"))

        # 📍 Convert location safely
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except ValueError:
            flash("Invalid location!", "danger")
            return redirect(url_for("register_user"))

        # 🔒 CHECK EMAIL (MAIN FIX)
        existing_user = User.query.filter_by(email=email).first()
        if existing_user is not None:
            flash("⚠️ Email already registered!", "warning")
            return redirect(url_for("register_user"))

        try:
            # 💾 CREATE USER
            new_user = User(
                name=name.strip(),
                email=email.strip().lower(),
                password=password,
                latitude=latitude,
                longitude=longitude
            )

            db.session.add(new_user)
            db.session.commit()

            flash("✅ Registration successful! Please login.", "success")
            return redirect("/login")

        # 🔥 HANDLE DUPLICATE ERROR (SAFETY NET)
        except IntegrityError:
            db.session.rollback()
            flash("⚠️ Email already exists!", "danger")
            return redirect(url_for("register_user"))

        # 🔥 HANDLE ANY OTHER ERROR
        except Exception as e:
            db.session.rollback()
            print("ERROR:", e)  # helpful for debugging
            flash("Something went wrong. Try again!", "danger")
            return redirect(url_for("register_user"))

    return render_template("register_user.html")
#-------------------- EDIT USER PROFILE --------------------
@app.route("/edit_user", methods=["GET", "POST"])
def edit_user():

    if "user_id" not in session:
        return redirect("/login")

    user = User.query.get(session["user_id"])

    if request.method == "POST":

        user.name = request.form.get("name")
        user.email = request.form.get("email")

        # ✅ NEW LOCATION CODE
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        if latitude and longitude:
            user.latitude = float(latitude)
            user.longitude = float(longitude)

        db.session.commit()

        return redirect("/user_dashboard")

    return render_template("edit_user.html", user=user)


# -------------------- LOGIN --------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session.clear()
            session["user_id"] = user.id
            session["role"] = "user"
            return redirect("/user_dashboard")

        worker = Worker.query.filter_by(email=email, password=password).first()
        if worker:
            session.clear()
            session["worker_id"] = worker.id
            session["role"] = "worker"
            return redirect("/worker_dashboard")

        return "Invalid Credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# -------------------- WORKER REGISTRATION --------------------

from sqlalchemy.exc import IntegrityError
from flask import flash, redirect, render_template, request

@app.route("/register_worker", methods=["GET", "POST"])
def register_worker():

    if request.method == "POST":

        # 📌 Get form data
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        password = request.form.get("password")
        service_type = request.form.get("service_type")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        # 🔍 BASIC VALIDATION
        if not first_name or not last_name or not email or not password or not service_type:
            flash("All fields are required!", "danger")
            return redirect("/register_worker")

        # 📍 LOCATION CHECK
        if not latitude or not longitude:
            flash("Please select location from map!", "danger")
            return redirect("/register_worker")

        # 📍 Convert location
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except:
            flash("Invalid location data!", "danger")
            return redirect("/register_worker")

        # 🔒 CHECK EMAIL EXISTS (FAST CHECK)
        existing_worker = Worker.query.filter_by(email=email).first()
        if existing_worker:
            flash("Email already registered!", "warning")
            return redirect("/register_worker")

        try:
            # 💾 Create new worker
            new_worker = Worker(
                first_name=first_name,
                last_name=last_name,
                email=email,
                password=password,
                service_type=service_type,
                latitude=latitude,
                longitude=longitude,
                works_done=0
            )

            db.session.add(new_worker)
            db.session.commit()

            flash("Registration successful! Please login.", "success")
            return redirect("/login")

        except IntegrityError:
            db.session.rollback()
            flash("Email already exists!", "danger")
            return redirect("/register_worker")

        except Exception as e:
            db.session.rollback()
            flash("Something went wrong. Try again.", "danger")
            return redirect("/register_worker")

    return render_template("register_worker.html")

# ------------ edit profile worker ------------

@app.route('/edit_worker', methods=['GET','POST'])
def edit_worker():

    if 'worker_id' not in session:
        return redirect('/login')

    worker = Worker.query.get(session['worker_id'])

    if request.method == 'POST':
        worker.first_name = request.form['first_name']
        worker.last_name = request.form['last_name']
        worker.service_type = request.form['service_type']
        worker.description = request.form['description']

        # 🧠 NEW LOCATION UPDATE
        worker.latitude = float(request.form['latitude'])
        worker.longitude = float(request.form['longitude'])

        db.session.commit()

        return redirect('/worker_dashboard')

    return render_template('edit_worker.html', worker=worker)

# -------------------- KNN RECOMMENDATION --------------------

def get_recommended_workers(service_type):

    workers = Worker.query.filter_by(service_type=service_type).all()

    if not workers:
        return []

    data = []

    for worker in workers:

        reviews = Review.query.filter_by(worker_id=worker.id).all()

        if reviews:
            avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 1)
            review_count = len(reviews)
        else:
            avg_rating = 0
            review_count = 0

        # attach to worker object (IMPORTANT)
        worker.avg_rating = avg_rating
        worker.review_count = review_count

        data.append([
            worker.id,
            worker.works_done,
            avg_rating,
            review_count
        ])

    df = pd.DataFrame(data, columns=[
        "id",
        "works_done",
        "avg_rating",
        "review_count"
    ])

    X = df[["works_done", "avg_rating", "review_count"]]

    knn = NearestNeighbors(n_neighbors=min(3, len(df)))
    knn.fit(X)

    best_worker = X.max().values.reshape(1,-1)

    distances, indices = knn.kneighbors(best_worker)

    recommended_ids = df.iloc[indices[0]]["id"].tolist()

    return Worker.query.filter(Worker.id.in_(recommended_ids)).all()

#-------------- search --------------
    
@app.route("/search", methods=["GET", "POST"])
def search():

    if session.get("role") != "user":
        return redirect("/login")

    # 🔥 STEP 1: GET SERVICES DYNAMICALLY (ALWAYS)
    services = db.session.query(Worker.service_type).distinct().all()
    services = [s[0] for s in services]

    workers_data = []

    if request.method == "POST":

        service_type = request.form.get("service_type")

        # Get workers
        if service_type:
            workers = Worker.query.filter_by(service_type=service_type).all()
        else:
            workers = Worker.query.all()

        if not workers:
            return render_template("search.html", workers=[], services=services)

        user = User.query.get(session["user_id"])

        data = []

        for worker in workers:

            reviews = Review.query.filter_by(worker_id=worker.id).all()

            if reviews:
                avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 1)
                review_count = len(reviews)

                toxic_count = 0
                for r in reviews:
                    is_toxic, _ = check_toxic(r.comment)
                    if is_toxic:
                        toxic_count += 1

                clean_score = round((review_count - toxic_count) / review_count, 2)

            else:
                avg_rating = 0
                review_count = 0
                clean_score = 0.8   # neutral

            # 📍 distance (only for badge + AI feature)
            distance = calculate_distance(
                user.latitude,
                user.longitude,
                worker.latitude,
                worker.longitude
            )

            data.append([
                worker.id,
                worker.first_name,
                worker.last_name,
                worker.service_type,
                worker.works_done,
                avg_rating,
                review_count,
                clean_score,
                distance
            ])

        df = pd.DataFrame(data, columns=[
            "id",
            "first_name",
            "last_name",
            "service",
            "works_done",
            "rating",
            "review_count",
            "clean_score",
            "distance"
        ])

        # ===============================
        # 🤖 KNN AI RANKING (CORE)
        # ===============================

        # Convert distance → score (closer = better)
        df["distance_score"] = 1 / (df["distance"] + 1)

        X = df[[
            "works_done",
            "rating",
            "review_count",
            "clean_score",
            "distance_score"
        ]]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        knn = NearestNeighbors(n_neighbors=len(df))
        knn.fit(X_scaled)

        ideal = X_scaled.max(axis=0).reshape(1, -1)

        distances, indices = knn.kneighbors(ideal)

        # 🔥 AI SORTING
        df = df.iloc[indices[0]].reset_index(drop=True)

        # ===============================
        # 🏷️ BADGE LOGIC
        # ===============================

        min_distance = df["distance"].min()

        df["badge"] = None

        for i in range(len(df)):

            if i == 0:
                df.loc[i, "badge"] = "ai_best"

            elif abs(df.loc[i, "distance"] - min_distance) < 0.01:
                df.loc[i, "badge"] = "closest"

            elif df.loc[i, "rating"] == 1:
                df.loc[i, "badge"] = "low"

        workers_data = df.to_dict(orient="records")

    # 🔥 IMPORTANT: always send services
    return render_template("search.html", workers=workers_data, services=services)
#--------------- chat list ------------- 

@app.route("/chat_list")
def chat_list():

    # 🔐 Only user access
    if session.get("role") != "user":
        return redirect("/login")

    user_id = session["user_id"]

    # 📩 Get all messages involving this user
    messages = Message.query.filter(
        (Message.sender_id == user_id) |
        (Message.receiver_id == user_id)
    ).all()

    # 🧠 Collect unique chat IDs
    chat_ids = set()

    for msg in messages:
        if msg.sender_id != user_id:
            chat_ids.add(msg.sender_id)
        if msg.receiver_id != user_id:
            chat_ids.add(msg.receiver_id)

    users = []

    for uid in chat_ids:

        # 🔍 Try User first
        u = User.query.get(uid)

        if u:
            name = u.name

        else:
            # 🔍 Try Worker
            w = Worker.query.get(uid)

            if not w:
                continue  # ❌ skip invalid IDs safely

            name = f"{w.first_name} {w.last_name}"

        # 📩 Get last message (latest)
        last_msg = Message.query.filter(
            ((Message.sender_id == user_id) & (Message.receiver_id == uid)) |
            ((Message.sender_id == uid) & (Message.receiver_id == user_id))
        ).order_by(Message.timestamp.desc()).first()

        # 🔴 Check unread messages
        unread_exists = Message.query.filter(
            Message.sender_id == uid,
            Message.receiver_id == user_id,
            Message.seen == False
        ).first()

        users.append({
            "id": uid,
            "name": name,
            "last_message": last_msg.content if last_msg else "",
            "last_time": last_msg.timestamp if last_msg else None,
            "unread": True if unread_exists else False
        })

    # 🔽 Safe sorting (handles None properly)
    users.sort(
        key=lambda x: x["last_time"] if x["last_time"] else 0,
        reverse=True
    )

    return render_template("chat_list.html", users=users)
# -------------------- HIRE --------------------

@app.route("/hire/<int:worker_id>")
def hire(worker_id):

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    # Check existing request
    existing_job = Job.query.filter(
        Job.user_id == user_id,
        Job.worker_id == worker_id,
        Job.status.in_(["Pending", "Hired"])
    ).first()

    if existing_job:
        flash("⚠️ You already have an active request with this worker.", "warning")
        return redirect("/search")

    # Create new job
    new_job = Job(
        user_id=user_id,
        worker_id=worker_id,
        status="Pending"
    )

    db.session.add(new_job)
    db.session.commit()

    flash("✅ Worker requested successfully!", "success")
    return redirect("/user_dashboard")

#------ worker jobs ------

@app.route("/worker_jobs")
def worker_jobs():

    if session.get("role") != "worker":
        return redirect("/login")

    worker_id = session.get("worker_id")

    # 🔹 Requested (Pending)
    pending_jobs = Job.query.filter_by(
        worker_id=worker_id,
        status="Pending"
    ).all()

    # 🔹 Active (Hired)
    active_jobs = Job.query.filter_by(
        worker_id=worker_id,
        status="Hired"
    ).all()

    # 🔹 Completed
    completed_jobs = Job.query.filter(
        Job.worker_id == worker_id,
        (Job.status == "Completed") | (Job.status == "Reviewed")
    ).all()

    # Attach user names (IMPORTANT for UI)
    def attach_user(jobs):
        result = []
        for job in jobs:
            user = User.query.get(job.user_id)
            result.append({
                "job": job,
                "user_name": user.name if user else "Unknown"
            })
        return result

    return render_template(
        "worker_jobs.html",
        pending_jobs=attach_user(pending_jobs),
        active_jobs=attach_user(active_jobs),
        completed_jobs=attach_user(completed_jobs)
    )


# -------------------- COMPLETE JOB --------------------

@app.route("/complete_job/<int:job_id>")
def complete_job(job_id):

    if session.get("role") != "user":
        return redirect("/login")

    job = Job.query.get(job_id)

    if job and job.status == "Hired":
        job.status = "Completed"

        worker = Worker.query.get(job.worker_id)
        if worker:
            worker.works_done += 1

        db.session.commit()

        # Redirect to review page
        return redirect(f"/give_review/{worker.id}")

    return redirect("/user_dashboard")

# -------------------- CANCEL JOB --------------------

@app.route("/cancel_job/<int:job_id>")
def cancel_job(job_id):
    if session.get("role") != "user":
        return redirect("/login")

    job = Job.query.get(job_id)
    if job:
        db.session.delete(job)
        db.session.commit()

    return redirect("/user_dashboard")


# -------------------- ADD REVIEW WITH TOXIC DETECTION --------------------

# -------------------- REVIEW --------------------

@app.route("/review/<int:worker_id>", methods=["POST"])
def review(worker_id):

    if session.get("role") != "user":
        return redirect("/login")

    # ⭐ NEW MULTI RATINGS
    quality = request.form.get("quality")
    punctuality = request.form.get("punctuality")
    professionalism = request.form.get("professionalism")
    reliability = request.form.get("reliability")

    comment = request.form.get("comment")

    if quality and punctuality and professionalism and reliability:

        # Convert to int
        quality = int(quality)
        punctuality = int(punctuality)
        professionalism = int(professionalism)
        reliability = int(reliability)

        # ⭐ AUTO CALCULATED OVERALL RATING
        rating = round(
            (quality + punctuality + professionalism + reliability) / 4, 1
        )

        # 🔥 TOXIC CHECK
        if comment:
            is_toxic, prob = check_toxic(comment)

            if is_toxic:
                comment = censor_text(comment)

        # SAVE REVIEW
        new_review = Review(
            worker_id=worker_id,
            quality=quality,
            punctuality=punctuality,
            professionalism=professionalism,
            reliability=reliability,
            rating=rating,
            comment=comment
        )

        db.session.add(new_review)

        # UPDATE JOB STATUS
        job = Job.query.filter_by(
            worker_id=worker_id,
            user_id=session["user_id"],
            status="Completed"
        ).first()

        if job:
            job.status = "Reviewed"

        db.session.commit()

    return redirect("/user_dashboard")
# -------------------- USER CHAT --------------------

@app.route("/chat/<int:worker_id>", methods=["GET", "POST"])
def chat(worker_id):

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    # mark worker messages as seen
    Message.query.filter_by(
        sender_id=worker_id,
        receiver_id=user_id,
        seen=False
    ).update({"seen": True})

    db.session.commit()

    # SEND MESSAGE
    if request.method == "POST":

        message_text = request.form.get("message")

        if message_text:

            new_message = Message(
                sender_id=user_id,
                receiver_id=worker_id,
                sender_type="user",
                content=message_text,
                seen=False
            )

            db.session.add(new_message)
            db.session.commit()

            return redirect(f"/chat/{worker_id}")

    # LOAD MESSAGES
    messages = Message.query.filter(
        (
            (Message.sender_id == user_id) &
            (Message.receiver_id == worker_id)
        ) |
        (
            (Message.sender_id == worker_id) &
            (Message.receiver_id == user_id)
        )
    ).order_by(Message.timestamp.asc()).all()

    worker = Worker.query.get(worker_id)

    return render_template(
        "chat.html",
        messages=messages,
        worker=worker
    )
# -------------------- WORKER CHAT --------------------

@app.route("/worker_chat/<int:user_id>", methods=["GET", "POST"])
def worker_chat(user_id):

    if session.get("role") != "worker":
        return redirect("/login")

    worker_id = session.get("worker_id")

    # mark messages as seen
    Message.query.filter_by(
        sender_id=user_id,
        receiver_id=worker_id,
        seen=False
    ).update({"seen": True})

    db.session.commit()
    user = User.query.get(user_id)
    worker = Worker.query.get(worker_id)

    # ---------------- SEND MESSAGE ----------------
    if request.method == "POST":

        message_text = request.form.get("message")

        if message_text:

            new_message = Message(
                sender_id=worker_id,
                receiver_id=user_id,
                sender_type="worker",
                content=message_text,
                seen=False
            )

            db.session.add(new_message)
            db.session.commit()

            return redirect(f"/worker_chat/{user_id}")
        

    # ---------------- LOAD MESSAGES ----------------
    messages = Message.query.filter(
        (
            (Message.sender_id == user_id) &
            (Message.receiver_id == worker_id)
        ) |
        (
            (Message.sender_id == worker_id) &
            (Message.receiver_id == user_id)
        )
    ).order_by(Message.timestamp.asc()).all()

    # ---------------- MARK AS SEEN ----------------
    for msg in messages:
        if msg.receiver_id == worker_id:
            msg.seen = True

    db.session.commit()

    # ---------------- RENDER PAGE ----------------
    return render_template(
        "worker_chat.html",
        user=user,
        worker=worker,
        messages=messages
    )
# -------------------- USER DASHBOARD --------------------

@app.route("/user_dashboard")
def user_dashboard():

    # Check login
    if session.get("role") != "user":
        return redirect("/login")

    user_id = session.get("user_id")

    # -------------------------
    # Get user
    # -------------------------
    user = User.query.get(user_id)

    # -------------------------
    # Get all jobs of user
    # -------------------------
    jobs = Job.query.filter_by(user_id=user_id).all()

    requested_jobs = []
    active_jobs = []
    completed_jobs = []

    for job in jobs:

        worker = Worker.query.get(job.worker_id)

        if not worker:
            continue

        job_data = {
            "job_id": job.id,
            "worker_id": worker.id,
            "worker_name": worker.first_name + " " + worker.last_name,
            "service": worker.service_type,
            "status": job.status
        }

        # -------------------------
        # Sort by status
        # -------------------------

        if job.status in ["Pending", "Requested"]:
            requested_jobs.append(job_data)

        elif job.status in ["Hired", "Accepted"]:
            active_jobs.append(job_data)

        elif job.status in ["Completed", "Reviewed"]:
            completed_jobs.append(job_data)

    # -------------------------
    # Reviews given by user
    # -------------------------
    reviews = Review.query.join(Job, Review.worker_id == Job.worker_id)\
        .filter(Job.user_id == user_id).all()

    # -------------------------
    # Chat conversations
    # -------------------------
    messages = Message.query.filter(
        (Message.sender_id == user_id) |
        (Message.receiver_id == user_id)
    ).all()

    worker_ids = set()

    for msg in messages:

        if msg.sender_type == "user":
            worker_ids.add(msg.receiver_id)

        else:
            worker_ids.add(msg.sender_id)

    workers = []

    for wid in worker_ids:

        worker = Worker.query.get(wid)

        if worker:
            workers.append(worker)

    # -------------------------
    # Render dashboard
    # -------------------------
    return render_template(
        "user_dashboard.html",
        user=user,
        requested_jobs=requested_jobs,
        active_jobs=active_jobs,
        completed_jobs=completed_jobs,
        reviews=reviews,
        workers=workers
    )

# -------------------- WORKER DASHBOARD --------------------

@app.route("/worker_dashboard")
def worker_dashboard():

    if session.get("role") != "worker":
        return redirect("/login")

    worker_id = session["worker_id"]

    worker = Worker.query.get(worker_id)

    # Reviews
    reviews = Review.query.filter_by(worker_id=worker_id).all()

    # Pending requests
    pending_jobs = Job.query.filter_by(worker_id=worker_id, status="Pending").all()

    # Accepted jobs
    hired_jobs = Job.query.filter_by(worker_id=worker_id, status="Hired").all()

    # Add user names to jobs
    pending_jobs_data = []
    for job in pending_jobs:
        user = User.query.get(job.user_id)
        pending_jobs_data.append({
            "job": job,
            "user_name": user.name if user else "Unknown"
        })

    hired_jobs_data = []
    for job in hired_jobs:
        user = User.query.get(job.user_id)
        hired_jobs_data.append({
            "job": job,
            "user_name": user.name if user else "Unknown"
        })

    # Chat users
    received_messages = Message.query.filter_by(receiver_id=worker_id).all()

    user_ids = set(msg.sender_id for msg in received_messages)

    users = []
    for uid in user_ids:
        user = User.query.get(uid)
        if user:
            users.append(user)

    return render_template(
        "worker_dashboard.html",
        worker=worker,
        reviews=reviews,
        users=users,
        pending_jobs=pending_jobs_data,
        hired_jobs=hired_jobs_data
    )
# -------------------- WORKER PROFILE --------------------
@app.route("/worker_profile/<int:worker_id>")
def worker_profile(worker_id):

    worker = Worker.query.get(worker_id)

    if not worker:
        return "Worker not found"

    # Get all jobs
    jobs = Job.query.filter_by(worker_id=worker_id).all()

    works_completed = 0

    for job in jobs:
        if job.status in ["Completed", "Reviewed"]:
            works_completed += 1

    # Get reviews
    reviews = Review.query.filter_by(worker_id=worker_id).all()

    # Calculate rating
    rating = 0

    if reviews:
        total = 0
        for r in reviews:
            total += r.rating
        rating = round(total / len(reviews), 1)

    return render_template(
        "worker_profile.html",
        worker=worker,
        reviews=reviews,
        works_completed=works_completed,
        rating=rating
    )
# ------- booking details --------------------
@app.route("/booking/<int:job_id>")
def booking_details(job_id):

    if session.get("role") != "user":
        return redirect("/login")

    job = Job.query.get(job_id)

    if not job or job.user_id != session["user_id"]:
        return redirect("/user_dashboard")

    worker = Worker.query.get(job.worker_id)

    return render_template("booking_details.html",
                           job=job,
                           worker=worker)
# -------------------- REVIEW PAGE --------------------
@app.route("/give_review/<int:worker_id>")
def give_review(worker_id):

    if session.get("role") != "user":
        return redirect("/login")

    worker = Worker.query.get(worker_id)

    return render_template("give_review.html", worker=worker)
#-------------------- ACCEPT JOB --------------------
@app.route("/accept_job/<int:job_id>")
def accept_job(job_id):

    job = Job.query.get(job_id)

    if job:
        job.status = "Hired"
        db.session.commit()

    return redirect("/worker_dashboard")
#-------------------- REJECT JOB --------------------

@app.route("/reject_job/<int:job_id>")
def reject_job(job_id):

    job = Job.query.get(job_id)

    if job:
        job.status = "Rejected"
        db.session.commit()

    return redirect("/worker_dashboard")
#----------- clickable services ------- 
@app.route("/service/<service_type>")
def service_workers(service_type):

    workers = Worker.query.filter_by(service_type=service_type).all()

    worker_list = []

    for worker in workers:

        reviews = Review.query.filter_by(worker_id=worker.id).all()

        if reviews:
            avg_rating = sum(r.rating for r in reviews) / len(reviews)
        else:
            avg_rating = 0

        worker_list.append({
            "id": worker.id,
            "name": worker.first_name + " " + worker.last_name,
            "service": worker.service_type,
            "works_done": worker.works_done,
            "rating": round(avg_rating, 1)
        })

    # 🔥 Sort workers (IMPORTANT)
    worker_list.sort(
        key=lambda x: (x["rating"], x["works_done"]),
        reverse=True
    )

    return render_template(
        "service_workers.html",
        workers=worker_list,
        service=service_type
    )

# -------------------- RUN --------------------

if __name__ == "__main__":
    app.run(debug=True)
