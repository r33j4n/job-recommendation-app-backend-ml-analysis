from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from get_cv_upload_response import query_ragcv,chat
from extract_details import populate_dbcv, clear_vector_db
from flask_cors import CORS
import os
import uuid
import json
from sql_chat import generate_and_run_query

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes from http://localhost:3000
UPLOAD_FOLDER = 'cv-library'



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, timeout=10000)

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file = request.files['file']
        user_id = request.form['userID']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        if not user_id:
            return jsonify({'error': 'No user ID provided'}), 400

        app.logger.info(f"Received file: {file.filename} from user: {user_id}")

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}_{user_id}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        clear_vector_db()

        # Populate the vector DB with the new CV
        documents = populate_dbcv([file_path])
        add_db_message = documents[1]

        # Remove the file after successful processing
        os.remove(file_path)

        return jsonify({'success': 'File successfully uploaded', 'db_message': add_db_message}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query-cv', methods=['POST'])
def query_cv():
    try:
        response = query_ragcv()
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-db', methods=['GET'])
def clear_db():
    try:
        clear_vector_db()
        return jsonify({'success': 'Vector DB cleared'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    print(question)

    if not question:
        return jsonify({"error": "Question parameter is required"}), 400

    try:
        result = chat(question)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


