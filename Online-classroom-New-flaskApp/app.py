# app.py
from flask import Flask
from app.config import Config
from app.db import mongo
from app.routes.warnings_route import warnings_bp
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object(Config)
# CORS(app, resources={r"/process_frame": {"origins": "http://localhost:5173"}}) #remove because defined cors in blueprint

# Enable CORS for the entire app (if needed, you can also specify which origins are allowed)
CORS(app, origins="http://localhost:5174")

# Initialize MongoDB
mongo.init_app(app)

# Register the warnings blueprint
app.register_blueprint(warnings_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True,port=5001)