
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, get_jwt, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from db import mongo
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json  # Import json for serialization
from bson import ObjectId
from datetime import timedelta

auth_controller = Blueprint('auth', __name__)

# Load environment variables from .env file
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@auth_controller.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    password = request.form.get("password")
    email = request.form.get("email")
    role = request.form.get("role", "student")  # Default to "student"

    if mongo.db.users.find_one({"username": username}):
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(password)

    profile_picture_url = None  # Initialize to None

    # Handle profile picture upload to Cloudinary
    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and allowed_file(file.filename):
            try:
                upload_result = cloudinary.uploader.upload(file)
                profile_picture_url = upload_result['secure_url']  # Get the secure URL
            except Exception as e:
                print(f"Error uploading to Cloudinary: {e}")
                return jsonify({"message": "Error uploading profile picture to Cloudinary."}), 500

    user_data = {
        "username": username,
        "password": hashed_password,
        "email": email,
        "role": role,
        "profile_picture": profile_picture_url,  # Store the URL in the database
    }

    try:
        mongo.db.users.insert_one(user_data)
    except Exception as e:
        print(f"Error inserting data into database: {e}")
        return jsonify({"message": "Error saving user to database."}), 500

    return jsonify({"message": "User registered successfully"}), 201


@auth_controller.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")  # Get the role from the request

    user = mongo.db.users.find_one({"username": username})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid username or password"}), 401

    if user["role"] != role: # Check the user role from db and request role are same or not
        return jsonify({"message": "Incorect role login"}), 401
    

    user_id = str(user["_id"])  # Convert ObjectId to string
    image = str(user["profile_picture"]) #profile img

    # Configure access token expiration to 24 hours
    access_token_expires = timedelta(hours=24)


    # Store only `user_id` as identity and add extra details in custom claims
    access_token = create_access_token(
        identity=user_id,  # Identity must be a string
        additional_claims={"username": username, "role": user["role"]},
        expires_delta=timedelta(hours=1)
    )
    
    return jsonify({"access_token": access_token, "user_id": user_id,"image":image,"username": username}), 200


@auth_controller.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    user_id = get_jwt_identity()  # This is now just the user_id
    claims = get_jwt()  # Retrieve additional claims (username, role)

    return jsonify({
        "message": f"Hello, {claims['username']}! You have access as a {claims['role']}.",
        "user_id": user_id
    }), 200


@auth_controller.route("/user/<user_id>", methods=["GET"])
@jwt_required()
def get_user_by_id(user_id):
    try:
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"password": 0})  # Exclude password
        if not user:
            return jsonify({"message": "User not found"}), 404

        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        return jsonify(user), 200

    except Exception as e:
        return jsonify({"message": "Invalid user ID", "error": str(e)}), 400
    

    