from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from db import mongo
import datetime
import base64
import time

frame_controller = Blueprint('frame', __name__)

# Parameters for loading data and images
detection_model_path = 'haarcascade/haarcascade_frontalface_default.xml'
emotion_model_path = 'pretrained_models/cnn.hdf5'

# Loading models
try:
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Cache to store the last update time for each student
student_last_update = {}
# Time interval for emotion updates (in seconds)
UPDATE_INTERVAL = 1  # Update every 1 second

@frame_controller.route('/process_frame', methods=['POST'])
@jwt_required()
def process_frame():
    data = request.get_json()
    print(f"Received data: {data}")
    frame_data = data.get('frame')
    student_id = data.get('student_id')
    username = data.get('username')

    if not frame_data or not student_id or not username:
        return jsonify({"error": "Missing required fields: frame, student_id, or username"}), 422

    # Use authenticated user ID if student_id not provided
    if not student_id:
        student_id = get_jwt_identity()

    if not frame_data:
        return jsonify({"error": "No frame data received"}), 400

    try:
        # Decode the base64 image
        img_data = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Resize the image
        img = cv2.resize(img, (800, 600))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        warnings = []
        detected_emotion = None
        should_update = False
        current_time = time.time()

        if len(faces) > 0:
            for (X, Y, W, H) in faces:
                # Extract face and predict emotion
                facial = gray[Y:Y + H, X:X + W]
                facial = cv2.resize(facial, (64, 64))
                facial = facial.astype("float") / 255.0
                facial = img_to_array(facial)
                facial = np.expand_dims(facial, axis=0)

                preds = emotion_classifier.predict(facial)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                detected_emotion = label

                # Check if the student is too far from the camera
                if W < 100 or H < 100:
                    warnings.append("Move closer to the camera")

                # Check if the student seems distracted
                if label in ["angry", "disgust", "scared", "sad"]:
                    warnings.append("Try to focus and manage your stress")
                elif label == "surprised":
                    warnings.append("Looks like you are surprised. Please focus on the content!")

                # Check if we should update based on time interval
                if student_id not in student_last_update:
                    should_update = True
                else:
                    # Calculate time difference
                    time_diff = current_time - student_last_update[student_id]['time']
                    last_emotion = student_last_update[student_id]['emotion']

                    # Update if enough time has passed or emotion has changed
                    if time_diff >= UPDATE_INTERVAL or detected_emotion != last_emotion:
                        should_update = True

                if should_update and detected_emotion:
                    # Update the last update time and emotion
                    student_last_update[student_id] = {
                        'time': current_time,
                        'emotion': detected_emotion
                    }

                    # Store the emotion in MongoDB with the username
                    mongo.db.student_emotions.insert_one({
                        'student_id': student_id,
                        'username': username,  # Store the username along with emotion
                        'emotion': detected_emotion,
                        'timestamp': datetime.datetime.utcnow()
                    })

                    # Emit the emotion data to connected clients
                    try:
                        from app import sio
                        sio.emit('emotion_data', {
                            'student_id': student_id,
                            'username': username,  # Include username in the emitted data
                            'emotion': detected_emotion,
                            'timestamp': datetime.datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        print(f"Error emitting socket event: {e}")

                break  # Only process the first detected face

            if not warnings:
                warnings.append("No Warning")
        else:
            warnings.append("No face detected")

        # Convert the processed image to base64 for display
        _, buffer = cv2.imencode('.jpg', img)
        processed_frame = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'processed_frame': processed_frame,
            'warnings': warnings,
            'emotion': detected_emotion,
            'updated_database': should_update
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500


@frame_controller.route('/student_emotions/<student_id>', methods=['GET'])
@jwt_required()
def get_student_emotions(student_id):
    """Endpoint to retrieve emotion history for a student"""
    try:
        # Get pagination parameters
        limit = int(request.args.get('limit', 50))
        skip = int(request.args.get('skip', 0))

        # Get emotions from the database
        emotions = list(mongo.db.student_emotions.find(
            {'student_id': student_id},
            {'_id': 0, 'emotion': 1, 'timestamp': 1}
        ).sort('timestamp', -1).skip(skip).limit(limit))

        # Format timestamps for JSON
        for emotion in emotions:
            if isinstance(emotion['timestamp'], datetime.datetime):
                emotion['timestamp'] = emotion['timestamp'].isoformat()

        return jsonify({
            'student_id': student_id,
            'emotions': emotions,
            'count': len(emotions)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@frame_controller.route('/clear_student_emotions/<student_id>', methods=['DELETE'])
@jwt_required()
def clear_student_emotions(student_id):
    """Endpoint to clear emotion history for a student"""
    try:
        result = mongo.db.student_emotions.delete_many({'student_id': student_id})

        # Also clear the cache entry
        if student_id in student_last_update:
            del student_last_update[student_id]

        return jsonify({
            'student_id': student_id,
            'deleted_count': result.deleted_count,
            'success': True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# frame.py

# ... (previous imports and code) ...

@frame_controller.route('/all_student_emotions', methods=['GET'])
@jwt_required()
def get_all_student_emotions():
    """Endpoint to retrieve the latest emotion for all students."""
    try:
        emotions = []
        for student in mongo.db.users.find({"role": "student"}):  # get all students

            student_id = str(student["_id"])  # convert ObjectId to String
            student_username = student["username"]
            emotion_data = mongo.db.student_emotions.find(
                {'student_id': student_id},
                {'_id': 0, 'emotion': 1, 'timestamp': 1}
            ).sort('timestamp', -1).limit(1)  # Get the latest emotion

            emotion = list(emotion_data)

            if emotion:
                emotion = emotion[0]
                if isinstance(emotion['timestamp'], datetime.datetime):
                    emotion['timestamp'] = emotion['timestamp'].isoformat()  # format time
                emotions.append({
                    'student_id': student_id,
                    'username': student_username,
                    'emotion': emotion.get('emotion'),  # Ensure emotion exists
                    'timestamp': emotion.get('timestamp')
                })
            else:
                emotions.append({
                    'student_id': student_id,
                    'username': student_username,
                    'emotion': "Unknown",
                    'timestamp': None
                })

        return jsonify(emotions), 200



    except Exception as e:
        print(f"Error retrieving all student emotions: {e}")
        return jsonify({"error": str(e)}), 500