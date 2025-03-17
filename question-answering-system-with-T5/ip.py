import os
import uuid
import torch
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pdfplumber
from ollama import chat
from prompts import get_mcq_prompt
from groq import Groq
from dotenv import load_dotenv
from pymongo import MongoClient
import random
from bson import json_util  # Import for ObjectId serialization

# MongoDB Configuration
client = MongoClient("mongodb+srv://edusmart:13579@edu-smart.s7iq2.mongodb.net")
db = client["myDatabase"]
collection = db["qa_collection"]

load_dotenv()
api_key = os.environ.get("Groq_Api_Key")

app = Flask(__name__)
CORS(app)


model = T5ForConditionalGeneration.from_pretrained("./t5-qna")
tokenizer = T5Tokenizer.from_pretrained("./t5-qna")

client = Groq(
    api_key=os.environ.get(api_key),
)


def predict_using_llama_api_v1(chunk, question):
    try:
        messages = get_mcq_prompt(chunk, question)

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            # print(content, end="")
            response += content

        return response

    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}


def predict_using_ollama(chunk, question):
    try:
        messages = get_mcq_prompt(chunk, question)

        stream = chat(model='llama2', messages=messages, stream=True)
        response = ""
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                response += content

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response "}

    except Exception as e:
        return {"error": f"Error generating : {str(e)}"}


def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise


def generate_questions(chunk, model, tokenizer):
    try:
        input_text = f"Given the following text, generate a concise and relevant question: {chunk}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)

        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not question or question.lower() in chunk.lower():
            question = "Could not generate a relevant question."

        return question
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise


def predict(context, query, model, tokenizer):
    try:
        input_text = f"Given the context: {context}, and the question: {query}, generate a precise and relevant answer."
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=8, early_stopping=True)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer.lower() == query.lower() or not answer:
            answer = "The answer is unclear, please try again."

        return answer
    except Exception as e:
        print(f"Error predicting answer: {str(e)}")
        raise


def process_pdf_and_generate_questions_with_context_stream(pdf_path, model, tokenizer, max_context_length=2048):
    try:
        text = extract_text_from_pdf(pdf_path)

        if not text:
            raise ValueError("No text extracted from the PDF. The file may be empty or unsupported.")

        chunks = [text[i:i + max_context_length] for i in range(0, len(text), max_context_length)]

        for i, chunk in enumerate(chunks[:10]):
            question = generate_questions(chunk, model, tokenizer)

            answer = predict(chunk, question, model, tokenizer)

            # print("answer", answer)
            print("question", question)

            refined_answer = predict_using_llama_api_v1(chunk, question)

            yield refined_answer

    except Exception as e:
        print(f"Error processing PDF and generating QA: {e}")
        yield {"error": str(e)}


@app.route("/generate-qa", methods=["POST"])
def generate_qa():
    file_path = None

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    unique_filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join("./", unique_filename)

    try:
        file.save(file_path)

        if not os.path.exists(file_path):
            return jsonify({"error": "Failed to save file"}), 500

        def generate():
            try:
                for qa_pair in process_pdf_and_generate_questions_with_context_stream(file_path, model, tokenizer):
                    yield f"data: {json.dumps(qa_pair)}\n\n"

                yield "data: {\"status\": \"complete\"}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"Failed to process PDF and generate QA\"}}\n\n"
            finally:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as cleanup_error:
                        print(f"Error during cleanup: {cleanup_error}")

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        print(f"Error in /generate-qa route: {str(e)}")
        return jsonify({"error": "An error occurred while processing the file."}), 500



# @app.route("/get-qa", methods=["GET"])
# def get_qa():
#     try:
#         # Fetch only quizzes where completed is False
#         quizzes = list(collection.find({"completed": False})) #THIS IS THE MOST IMPORTANT PART

#         # Convert ObjectId to string and properly serialize
#         return json.dumps(quizzes, default=json_util.default), 200
#     except Exception as e:
#         print(f"Error fetching quizzes: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route("/questions", methods=["GET"])
def get_all_questions():
    """
    Retrieves all questions from the qa_collection in MongoDB.
    """
    try:
        questions = list(qa_collection.find({}))  # Fetch all documents
        return Response(json_util.dumps(questions, default=json_util.default), mimetype='application/json'), 200
    except Exception as e:
        logging.exception("Error fetching all questions.")
        return jsonify({"error": str(e)}), 500

def save_exam_report(user_id, exam_id, score):
    try:
        report_data = {
            "user_id": user_id,
            "exam_id": exam_id,
            "score": score,
            "timestamp": datetime.datetime.now()
        }
        result = report_collection.insert_one(report_data)
        return str(result.inserted_id)  # Return the inserted report ID as string
    except Exception as e:
        print(f"Error saving exam report: {e}")
        return None

# Helper function to retrieve exam reports for a user
def get_exam_reports_for_user(user_id):
    try:
        # Find the reports and covert them to proper JSON format
        reports = list(report_collection.find({"user_id": user_id}))
        return json.dumps(reports, default=json_util.default)
    except Exception as e:
        print(f"Error fetching exam reports: {e}")
        return []

# Backend to retrieve Exam from the front end
@app.route("/get-qa/<exam_id>", methods=["GET"])
def get_qa_by_id(exam_id):
    try:
        quiz = qa_collection.find_one({"_id": ObjectId(exam_id)})

        if quiz:
            return jsonify(quiz), 200
        else:
            return jsonify({"error": "Quiz not found"}), 404
    except Exception as e:
        print(f"Error fetching quiz: {e}")
        return jsonify({"error": str(e)}), 500

# Assuming the following to make it accurate
@app.route("/submit-exam", methods=["POST"])
def submit_exam():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_id = data.get("userId")
    exam_id = data.get("examId")
    score = data.get("score")
    print("Recieved all the following data:", {user_id, exam_id, score})  # Check if they are correct

    if not all([user_id, exam_id, score]):
        return jsonify({"error": "Missing required data"}), 400

    # Save the report
    report_id = save_exam_report(user_id, exam_id, score)
    if report_id:
        return jsonify({"message": "Exam submitted successfully!", "report_id": report_id}), 200
    else:
        return jsonify({"error": "Failed to save exam report"}), 500

# Helper function to serialize data
def serialize_data(data):
    try:
        return json.dumps(data, default=json_util.default)
    except Exception as e:
        print(f"Error serializing data: {e}")
        return None

# API route to get exam reports
@app.route("/get-exam-reports/<user_id>", methods=["GET"])
def get_exam_reports(user_id):
    reports = get_exam_reports_for_user(user_id)
    if reports:
        return jsonify(reports), 200
    else:
        return jsonify({"message": "No exam reports found for this user"}), 404

@app.route("/test-groq", methods=["GET"])
def test_groq():
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "how are you"
                },
                {
                    "role": "assistant",
                    "content": "I'm just a language model, so I don't have emotions or physical sensations like humans do. However, I'm functioning properly and ready to assist you with any questions or tasks you may have. How can I help you today?"
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response_content = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            response_content += content

        return jsonify({"response": response_content}), 200
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)