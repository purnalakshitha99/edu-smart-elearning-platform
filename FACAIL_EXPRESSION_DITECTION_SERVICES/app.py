from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, verify_jwt_in_request
from db import mongo  # Import the mongo object
from routes.frame import frame_controller
from config import Config
from flask_socketio import SocketIO
import eventlet

# Use SocketIO from flask_socketio
sio = SocketIO(cors_allowed_origins='*', async_mode='eventlet')

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask extensions here
    CORS(app)  # Enable CORS for all routes
    JWTManager(app)

    # Initialize MongoDB
    mongo.init_app(app)  # Initialize the mongo object with the Flask app

    # Register blueprints
    app.register_blueprint(frame_controller, url_prefix="/api")

    # Initialize SocketIO
    sio.init_app(app, cors_allowed_origins='*')

    return app

app = create_app()

# SocketIO event handlers
@sio.on('connect')
def handle_connect(sid, environ):
    # Fetch the token from query params
    token = request.args.get('auth')
    if not token:
        # If token is missing, deny the connection
        return False  # This will reject the connection

    try:
        # Validate the token using flask_jwt_extended
        # `verify_jwt_in_request` is used for verifying JWT token
        verify_jwt_in_request()  # Will throw an exception if the token is invalid
        print(f'Client {sid} connected with valid token.')
    except Exception as e:
        print(f'Connection rejected for {sid}: {str(e)}')
        return False  # Reject the connection if token is invalid
@sio.on('disconnect')
def disconnect(sid):
    print('Client disconnected:', sid)

# Test MongoDB connection
@app.route('/test_mongo', methods=['GET'])
def test_mongo():
    try:
        # Check if the database is accessible
        mongo.db.command('ping')
        return jsonify({"message": "Connected to MongoDB successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to connect to MongoDB: {str(e)}"}), 500

if __name__ == '__main__':
     eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5003)), app)
