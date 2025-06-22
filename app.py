import os
import json
from datetime import datetime
import torch
from flask import Flask, render_template, request, jsonify, session, Response
import threading
import uuid

# Import the AIAssistant class from main.py
from main import AIAssistant

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global variable to store the model and assistant
global_assistant = None
model_loading_status = {"status": "not_started", "progress": 0, "message": ""}

def load_model_in_background():
    """Load the model in a background thread to avoid blocking the web server"""
    global global_assistant, model_loading_status
    try:
        model_loading_status = {"status": "loading", "progress": 10, "message": "Initializing AI Assistant..."}
        # Create the AI Assistant instance
        global_assistant = AIAssistant(name="SRUSTI_AI_ASSISTANT")
        model_loading_status = {"status": "complete", "progress": 100, "message": "Model loaded successfully!"}
    except Exception as e:
        model_loading_status = {"status": "error", "progress": 0, "message": f"Error loading model: {str(e)}"}
        print(f"Error loading model: {e}")

# Start loading the model in the background when the app starts
threading.Thread(target=load_model_in_background).start()

@app.route('/')
def index():
    """Render the main page"""
    # Generate a unique session ID if not present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    return render_template('index.html')

@app.route('/model_status')
def model_status():
    """Return the current status of model loading"""
    global model_loading_status
    return jsonify(model_loading_status)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a response based on the input"""
    global global_assistant
    if global_assistant is None:
        return jsonify({
            "success": False,
            "error": "Model is still loading. Please wait.",
            "model_status": model_loading_status
        })
    data = request.json
    prompt = data.get('prompt', '')
    function_type = data.get('function_type', 'question')  # question, summary, creative
    if not prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty"})
    try:
        # Use the appropriate method from AIAssistant based on function_type
        if function_type == "question":
            response = global_assistant.get_ai_response(prompt, "question")
        elif function_type == "summary":
            response = global_assistant.get_ai_response(prompt, "summary")
        elif function_type == "creative":
            response = global_assistant.get_ai_response(prompt, "creative")
        else:
            response = global_assistant.get_ai_response(prompt, "question")
        # Store in session conversation history
        if 'conversation_history' in session:
            session['conversation_history'].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": function_type,
                "prompt": prompt,
                "response": response
            })
            session.modified = True
        return jsonify({
            "success": True,
            "response": response,
            "function_type": function_type
        })
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({
            "success": False,
            "error": f"Error generating response: {str(e)}"
        })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Collect feedback on a response"""
    global global_assistant
    if global_assistant is None:
        return jsonify({"success": False, "error": "AI Assistant not initialized"})
    data = request.json
    function_type = data.get('function_type', '')
    prompt = data.get('prompt', '')
    response = data.get('response', '')
    rating = data.get('rating', 0)
    comments = data.get('comments', '')
    # Create feedback entry
    feedback_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "function_type": function_type,
        "prompt": prompt,
        "response": response,
        "rating": rating,
        "comments": comments
    }
    # Add to assistant's feedback data
    global_assistant.feedback_data.append(feedback_entry)
    global_assistant.save_feedback()
    return jsonify({
        "success": True,
        "message": "Thank you for your feedback!"
    })

@app.route('/history', methods=['GET'])
def history():
    """Get conversation history from the session"""
    if 'conversation_history' in session:
        return jsonify({
            "success": True,
            "history": session['conversation_history']
        })
    else:
        return jsonify({
            "success": False,
            "error": "No conversation history found"
        })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the conversation history from the session"""
    if 'conversation_history' in session:
        session['conversation_history'] = []
        session.modified = True
        return jsonify({
            "success": True,
            "message": "Conversation history cleared"
        })
    else:
        return jsonify({
            "success": False,
            "error": "No conversation history found"
        })

@app.route('/download_history', methods=['GET'])
def download_history():
    """Generate a downloadable JSON file of the conversation history"""
    if 'conversation_history' not in session or not session['conversation_history']:
        return jsonify({
            "success": False,
            "error": "No conversation history to download"
        })
    history_data = json.dumps(session['conversation_history'], indent=2)
    return Response(
        history_data,
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment;filename=conversation_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        }
    )

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": global_assistant is not None,
        "model_status": model_loading_status
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "The requested resource was not found"
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        "success": False,
        "error": "An internal server error occurred"
    }), 500

if __name__ == '__main__':
    # Check if CUDA is available and print info
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU mode (slower).")
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Check if index.html exists in templates directory
    if not os.path.exists('templates/index.html'):
        print("Warning: templates/index.html not found. Please make sure to create this file.")
        print("The application will still start, but you may see a 'TemplateNotFound' error when accessing the web interface.")
    # Run the Flask app
print("Starting AI Assistant web server...")
print("Access the web interface at http://localhost:5000")
app.run(debug=True, host='0.0.0.0', port=5000)