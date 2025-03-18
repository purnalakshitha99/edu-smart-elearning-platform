# config.py
class Config:
    # MongoDB URI for MongoDB Atlas
    MONGO_URI = "mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net/myDatabase?retryWrites=true&w=majority"
    SECRET_KEY = "your_secret_key_here"
    JWT_SECRET_KEY = "your-secret-key"
    # Optional, for Flask sessions if needed
