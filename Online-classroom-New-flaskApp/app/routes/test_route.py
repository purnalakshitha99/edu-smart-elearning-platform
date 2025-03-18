# routes/test_route.py
from flask import Blueprint, jsonify
from db import mongo  # Import mongo from db.py

# Initialize Flask Blueprint for testing
test_bp = Blueprint('test', __name__)


# Don't access mongo.db at the module level
# Instead, access it within the route functions

# Test route to insert a record into MongoDB
@test_bp.route('/test_insert', methods=['GET'])
def test_insert():
    try:
        # Access the collection inside the function
        warnings_collection = mongo.db.warnings

        # Insert a test document into the 'warnings' collection
        test_entry = {
            "student_id": "test_student_1",
            "warning_message": "Test warning",
            "timestamp": 1633373842  # Example timestamp
        }

        # Insert the test document into the collection
        warnings_collection.insert_one(test_entry)

        # Return a success message if inserted successfully
        return jsonify({"message": "Test record inserted successfully!"}), 200
    except Exception as e:
        # If any error occurs, return an error message
        return jsonify({"error": f"Error inserting test record: {e}"}), 500