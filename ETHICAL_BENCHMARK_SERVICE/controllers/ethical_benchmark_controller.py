from flask import Blueprint, jsonify, request, current_app
from bson.objectid import ObjectId
from pipe.ethical_benchmark_pipeline import EthicalBenchMarkDetector
from db import mongo

ethical_benchmark_controller = Blueprint('ethical_benchmark', __name__)

@ethical_benchmark_controller.route("/")
def video_feed():
    user_id = request.args.get("userid")  # Get userId from the request
    if not user_id:
        return jsonify(error="User ID is required"), 400

    app = current_app._get_current_object()
    detector = EthicalBenchMarkDetector(app, user_id)  # Pass userId to the detector
    detector.run()
    return jsonify(message="Ethical Benchmark started")

@ethical_benchmark_controller.route("/report/<user_id>", methods=['GET'])
def get_ethical_report(user_id):
    report = generate_ethical_report(user_id)
    return jsonify(report)

def generate_ethical_report(user_id):
    try:
        # Ensure user_id is a valid ObjectId
        user_object_id = ObjectId(user_id)

        # Query the database for cheating events specific to the user
        cheating_events = mongo.db.cheating_events.find({'user_id': user_object_id})

        report = []
        for event in cheating_events:
            # Convert ObjectId to string for JSON serialization
            event['_id'] = str(event['_id'])
            report.append(event)

        return report

    except Exception as e:
        print(f"Error generating report: {e}")
        return {"message": "Error generating report", "error": str(e)}