"""
EssayLens — Production Flask API
- Auth:     bcrypt passwords + JWT tokens
- Database: PostgreSQL via SQLAlchemy (Supabase free tier)
- Hosting:  Render free tier
- ML:       TensorFlow-free numpy inference

Environment variables to set in Render dashboard:
  DATABASE_URL   — from Supabase (postgres://...)
  JWT_SECRET_KEY — any long random string
  FRONTEND_URL   — your Netlify URL e.g. https://aidetector.netlify.app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from datetime import timedelta
import numpy as np
import pickle
import re
import os

# App setup
app = Flask(__name__)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}}, supports_credentials=True)

app.config["SQLALCHEMY_DATABASE_URI"]        = os.environ.get("DATABASE_URL", "sqlite:///local_dev.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"]                 = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"]       = timedelta(days=7)

db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt    = JWTManager(app)


# Models
class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    first_name    = db.Column(db.String(50),  nullable=False)
    last_name     = db.Column(db.String(50),  nullable=False)
    email         = db.Column(db.String(120), nullable=False, unique=True)
    password_hash = db.Column(db.String(200), nullable=False)
    role          = db.Column(db.String(20),  default="student")
    created_at    = db.Column(db.DateTime,    server_default=db.func.now())
    scans         = db.relationship("Scan", backref="user", lazy=True, cascade="all, delete")

    def to_dict(self):
        return {
            "id":        self.id,
            "firstName": self.first_name,
            "lastName":  self.last_name,
            "email":     self.email,
            "role":      self.role,
        }


class Scan(db.Model):
    __tablename__  = "scans"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    preview        = db.Column(db.String(300))
    verdict        = db.Column(db.String(20))
    label          = db.Column(db.Integer)
    ai_probability = db.Column(db.Float)
    word_count     = db.Column(db.Integer)
    created_at     = db.Column(db.DateTime, server_default=db.func.now())

    def to_dict(self):
        return {
            "id":           self.id,
            "preview":      self.preview,
            "verdict":      self.verdict,
            "label":        self.label,
            "pct":          round(self.ai_probability * 100),
            "word_count":   self.word_count,
            "date":         self.created_at.strftime("%b %-d, %Y") if self.created_at else "",
        }


# ML artifacts
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "tfidf_tokenizer.pkl")
WEIGHTS_PATH   = os.environ.get("WEIGHTS_PATH",   "model_weights.pkl")

tfidf_tokenizer = None
layer_params    = None

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

ACTIVATIONS = {
    "relu": relu, "sigmoid": sigmoid,
    "linear": lambda x: x, "tanh": np.tanh,
}

def load_artifacts():
    global tfidf_tokenizer, layer_params
    with open(TOKENIZER_PATH, "rb") as f:
        tfidf_tokenizer = pickle.load(f)
    print(f"Loaded tokenizer  → {TOKENIZER_PATH}")

    with open(WEIGHTS_PATH, "rb") as f:
        data = pickle.load(f)

    layer_params = []
    for cfg, w in zip(data["configs"], data["weights"]):
        if not w or len(w) != 2:
            continue
        W, b = w[0], w[1]
        act = cfg.get("activation", {})
        if isinstance(act, dict):
            act = act.get("class_name", "linear")
        layer_params.append((W, b, ACTIVATIONS.get(act, ACTIVATIONS["linear"])))
    print(f"Loaded model      → {WEIGHTS_PATH}  ({len(layer_params)} dense layers)")

def numpy_predict(sparse_features):
    x = sparse_features.toarray().astype(np.float32)
    for W, b, act in layer_params:
        x = act(x @ W + b)
    return float(x.ravel()[0])

def compute_signals(text):
    words          = text.split()
    sentences      = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 8]
    chars_no_space = text.replace(" ", "")
    unique_words   = len(set(w.lower().strip(".,!?\"'") for w in words))
    punct_count    = sum(1 for c in text if c in ',.;:—–()"\'')
    bigrams        = {words[i] + "_" + words[i+1] for i in range(len(words) - 1)}
    return {
        "word_count":       len(words),
        "vocab_richness":   round(unique_words / max(len(words), 1), 3),
        "avg_sentence_len": round(len(words) / max(len(sentences), 1), 1),
        "punct_density":    round(punct_count / max(len(chars_no_space), 1) * 100, 2),
        "bigram_diversity": round(len(bigrams) / max(len(words) - 1, 1), 3),
        "unique_words":     unique_words,
        "bigram_count":     len(bigrams),
    }


# Auth routes
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    first = (data.get("firstName") or "").strip()
    last  = (data.get("lastName")  or "").strip()
    email = (data.get("email")     or "").strip().lower()
    pw    = data.get("password", "")
    role  = data.get("role", "student")

    if not all([first, last, email, pw]):
        return jsonify({"error": "All fields are required."}), 400
    if len(pw) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists."}), 409

    hashed = bcrypt.generate_password_hash(pw).decode("utf-8")
    user   = User(first_name=first, last_name=last, email=email, password_hash=hashed, role=role)
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "user": user.to_dict()}), 201


@app.route("/login", methods=["POST"])
def login():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    pw    = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, pw):
        return jsonify({"error": "Incorrect email or password."}), 401

    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "user": user.to_dict()}), 200


@app.route("/me", methods=["GET"])
@jwt_required()
def me():
    user = User.query.get(get_jwt_identity())
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"user": user.to_dict()})


# Scan routes
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    if layer_params is None or tfidf_tokenizer is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 50:
        return jsonify({"error": "Text too short — provide at least 50 characters."}), 400

    try:
        features = tfidf_tokenizer.transform([text])
        ai_prob  = numpy_predict(features)
        label    = 1 if ai_prob >= 0.5 else 0
        signals  = compute_signals(text)
        words    = text.split()
        preview  = " ".join(words[:8]) + ("…" if len(words) > 8 else "")

        # Save to database
        scan = Scan(
            user_id        = get_jwt_identity(),
            preview        = preview,
            verdict        = "AI-Generated" if label == 1 else "Human Written",
            label          = label,
            ai_probability = round(ai_prob, 4),
            word_count     = signals["word_count"],
        )
        db.session.add(scan)
        db.session.commit()

        return jsonify({
            "label":          label,
            "verdict":        scan.verdict,
            "ai_probability": round(ai_prob, 4),
            "confidence":     round(max(ai_prob, 1 - ai_prob), 4),
            "signals":        signals,
            "scan_id":        scan.id,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scans", methods=["GET"])
@jwt_required()
def get_scans():
    scans = (Scan.query
             .filter_by(user_id=get_jwt_identity())
             .order_by(Scan.created_at.desc())
             .limit(50)
             .all())
    return jsonify({"scans": [s.to_dict() for s in scans]})


@app.route("/scans/<int:scan_id>", methods=["DELETE"])
@jwt_required()
def delete_scan(scan_id):
    scan = Scan.query.filter_by(id=scan_id, user_id=get_jwt_identity()).first()
    if not scan:
        return jsonify({"error": "Scan not found."}), 404
    db.session.delete(scan)
    db.session.commit()
    return jsonify({"deleted": scan_id})


# Health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     layer_params is not None,
        "tokenizer_loaded": tfidf_tokenizer is not None,
    })


# Entry point
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("Database tables ready")

    try:
        load_artifacts()
    except FileNotFoundError as e:
        print(f"\n[!] WARNING: {e}")
        print("    /predict returns 503 until model files are present.\n")

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)