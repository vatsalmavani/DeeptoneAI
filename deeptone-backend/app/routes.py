from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from datetime import datetime
import numpy as np

from app.classifier import classify_audio
from app.audio_classification import extract_features
from app.database import predictions_collection, users_collection

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return "Welcome to our deepfake voice detection system..."

#  AUDIO PREDICTION 
@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files['file']
    username = request.form.get('username')

    if not username:
        return jsonify({"success": False, "error": "Username is required"}), 400

    try:
        result = classify_audio(file)

        clean_result = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in result.items()
        }

        clean_result.update({
            "filename": file.filename,
            "timestamp": datetime.utcnow(),
            "username": username,
        })

        if predictions_collection:
            inserted = predictions_collection.insert_one(clean_result)
            clean_result["_id"] = str(inserted.inserted_id)

        return jsonify(clean_result), 200

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"success": False, "error": "Internal server error", "details": str(e)}), 500


#  REGISTER 
@main.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json() if request.is_json else request.form
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password are required"}), 400

        if users_collection.find_one({"username": username}):
            return jsonify({"success": False, "error": "User already exists"}), 409

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "username": username,
            "password": hashed_password
        })

        return jsonify({"success": True, "message": "User registered successfully", "username": username}), 201

    except Exception as e:
        print("Register error:", str(e))
        return jsonify({"success": False, "error": "Registration failed", "details": str(e)}), 500


# LOGIN 
@main.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json() if request.is_json else request.form
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400

        user = users_collection.find_one({"username": username})
        if not user:
            return jsonify({"success": False, "error": "Invalid username"}), 401

        if not check_password_hash(user['password'], password):
            return jsonify({"success": False, "error": "Invalid password"}), 401

        return jsonify({"success": True, "message": "Login successful", "username": username}), 200

    except Exception as e:
        print("Login error:", str(e))
        return jsonify({"success": False, "error": "Login failed", "details": str(e)}), 500


# HISTORY 
@main.route('/history/<username>', methods=['GET'])
def get_history(username):
    try:
        if not username:
            return jsonify({"success": False, "error": "Username required"}), 400

        if predictions_collection is None:
            return jsonify([]), 200

        history_cursor = predictions_collection.find(
            {"username": username}
        ).sort("timestamp", -1)

        history = []
        for item in history_cursor:
            history.append({
                "_id": str(item["_id"]),
                "prediction": item.get("prediction"),
                "accuracy": float(item.get("accuracy", 0)),
                "recall": float(item.get("recall", 0)),
                "precision": float(item.get("precision", 0)),
                "f1_score": float(item.get("f1_score", 0)),
                "filename": item.get("filename", "unknown"),
                "timestamp": item.get("timestamp").isoformat() if item.get("timestamp") else "",
                "confidence": float(item.get("confidence", 0))
            })

        return jsonify(history), 200

    except Exception as e:
        print("History error:", str(e))
        return jsonify({"success": False, "error": "Failed to fetch history", "details": str(e)}), 500
