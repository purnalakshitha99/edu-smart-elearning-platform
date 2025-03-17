# from flask import Flask, request, redirect, jsonify
# from static.logger import logging
# from controllers.auth_controller import auth_controller
# from controllers.ethical_benchmark_controller import ethical_benchmark_controller
# from controllers.test_controller import test_controller
# from flask_jwt_extended import JWTManager
# from db import mongo  # Import mongo from db.py
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# app.config['MONGO_URI'] = "mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net/myDatabase?retryWrites=true&w=majority"
# mongo.init_app(app)  # Initialize MongoDB with Flask app

# logging.info(f'Preprocessed Text : {"Flask Server is started"}')

# app.config["JWT_SECRET_KEY"] = "your-secret-key"  
# jwt = JWTManager(app)

# app.register_blueprint(auth_controller, url_prefix='/auth')
# app.register_blueprint(ethical_benchmark_controller, url_prefix='/ethical_benchmark')
# app.register_blueprint(test_controller, url_prefix='/test')

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask
from static.logger import logging
from controllers.auth_controller import auth_controller
from controllers.ethical_benchmark_controller import ethical_benchmark_controller
from controllers.test_controller import test_controller
from flask_jwt_extended import JWTManager
from db import mongo
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CORS(app, origins="http://localhost:5174")

app.config['MONGO_URI'] = "mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net/myDatabase?retryWrites=true&w=majority"
app.config["JWT_SECRET_KEY"] = "your-secret-key"

mongo.init_app(app)

logging.info(f'Preprocessed Text : {"Flask Server is started"}')

jwt = JWTManager(app)

app.register_blueprint(auth_controller, url_prefix='/auth')
app.register_blueprint(ethical_benchmark_controller, url_prefix='/ethical_benchmark')
app.register_blueprint(test_controller, url_prefix='/test')

if __name__ == "__main__":
    app.run(debug=True)