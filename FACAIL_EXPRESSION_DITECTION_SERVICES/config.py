# facerecognition/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    JWT_SECRET_KEY = "your-secret-key"    # Change this!
    MONGO_URI = "mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net/myDatabase?retryWrites=true&w=majority"