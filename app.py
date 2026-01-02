import os
import json
import tempfile
import logging
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response
import threading
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import time

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

from textblob import TextBlob
import nltk

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables
model = None
tokenizer = None
model_loading_status = {"status": "loading", "progress": 0, "message": "Initializing model..."}
feedback_data = []
feedback_file = "feedback_data.json"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_load_start_time = time.time()
sia = SentimentIntensityAnalyzer()





def load_feedback():
    """Load previous feedback if available"""
    global feedback_data
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
                logger.info(f"Loaded {len(feedback_data)} feedback entries")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading feedback: {e}")
        feedback_data = []

def save_feedback():
    """Save feedback to file"""
    try:
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving feedback: {e}")

def load_model_background():
    """Load model in background thread"""
    global model, tokenizer, model_loading_status, model_load_start_time
    model_load_start_time = time.time()
    try:
        model_loading_status = {"status": "loading", "progress": 25, "message": "Loading tokenizer..."}
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_loading_status = {"status": "loading", "progress": 50, "message": "Loading model weights..."}
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        model.eval()
        
        elapsed = time.time() - model_load_start_time
        model_loading_status = {"status": "complete", "progress": 100, "message": f"Model ready! (loaded in {elapsed:.1f}s)"}
        print(f"Model loaded successfully in {elapsed:.1f}s")
    except Exception as e:
        model_loading_status = {"status": "error", "progress": 0, "message": f"Error: {str(e)}"}
        print(f"Error loading model: {e}")

def sanitize_response(text):
    """Remove Markdown strikethrough from model output"""
    text = re.sub(r'~~', '', text)
    text = re.sub(r'</?del>', '', text)
    return text

def get_groq_response(prompt, function_type="question"):
    """Fast Groq API - requires free API key"""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return "Please set GROQ_API_KEY environment variable"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompts = {
        "question": "Answer this question concisely and accurately.",
        "summary": "Summarize this text in 2-3 sentences.",
        "creative": "Write creative content based on this request.",
        "paraphrase": "Rephrase this text while keeping the meaning."
    }
    
    data = {
        "model": "llama3-8b-8192",  # Very fast model
        "messages": [
            {"role": "system", "content": system_prompts.get(function_type, "Help with this request.")},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            return f"API Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Groq request failed: {str(e)}")
        return f"Request failed: {str(e)}"

def get_ollama_response(prompt, function_type="question"):
    """Local Ollama API - faster than transformers"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": "llama3.2:1b",  # Lightweight model
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1000
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=20)
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        else:
            return f"Ollama Error: {response.status_code}"
    except Exception as e:
        return f"Ollama not running: {str(e)}"

def get_huggingface_response(prompt, function_type="question"):
    """Hugging Face Inference API - free tier available"""
    api_key = os.environ.get('HF_API_KEY')
    if not api_key:
        return "Please set HF_API_KEY environment variable"
    
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    data = {"inputs": prompt, "parameters": {"max_length": 1000}}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response generated')
            return str(result)
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Request failed: {str(e)}"

def get_fast_response(prompt, function_type):
    """Try multiple APIs in order of speed until one works"""
    
    # Handle common questions with predefined responses
    prompt_lower = prompt.lower().strip()
    
    if prompt_lower in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']:
        return "Hello! I'm SRUSTI, your AI assistant. I'm here to help you with questions, creative writing, summaries, and more. What can I do for you today?"
    
    if prompt_lower in ['how are you', 'how are you?', 'how do you do', 'how are things']:
        return "I'm doing great, thank you for asking! I'm ready to help you with any questions or tasks you have. What would you like to work on today?"
    
    if prompt_lower in ['what can you do', 'what can you do?', 'help', 'what are your capabilities']:
        return "I can help you with: \n• Answering questions\n• Writing creative content (stories, poems, essays)\n• Summarizing text\n• Paraphrasing content\n• Analyzing sentiment\n• And much more! Just ask me anything."
    
    if len(prompt.strip()) < 3:
        return "I'd love to help! Could you please provide a bit more detail about what you'd like me to assist you with?"
    
    # Try APIs in order of speed (fastest first)
    apis = [
        ("Groq", get_groq_response),
        ("Ollama", get_ollama_response), 
        ("HuggingFace", get_huggingface_response)
    ]
    
    for api_name, api_func in apis:
        try:
            response = api_func(prompt, function_type)
            if response and not response.startswith(("Please set", "API Error", "Request failed", "Ollama not running")):
                logger.info(f"Using {api_name} API for response")
                return response
        except Exception as e:
            logger.warning(f"{api_name} API failed: {e}")
            continue
    
    # If all APIs fail, use local TinyLlama as fallback
    logger.info("All fast APIs failed, using local TinyLlama")
    return get_tinyllama_response_local(prompt, function_type)

def get_tinyllama_response_local(prompt, function_type):
    """Get response from TinyLlama model"""
    global model, tokenizer
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if model is None or tokenizer is None:
        elapsed = time.time() - model_load_start_time
        if elapsed > 300:
            return "Model loading timed out. Please restart the server."
        return f"Model is still loading. Please wait... ({elapsed:.0f}s elapsed)"
    
    try:
        if function_type == "question":
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            max_tokens = 2000
        elif function_type == "summary":
            formatted_prompt = f"Text: {prompt}\n\nSummary:"
            max_tokens = 1000
        elif function_type == "paraphrase":
            formatted_prompt = f"Original: {prompt}\n\nParaphrase:"
            max_tokens = 800
        elif function_type == "keypoints":
            formatted_prompt = f"Text: {prompt}\n\nKey points:\n-"
            max_tokens = 1000
        elif function_type == "simplify":
            formatted_prompt = f"Complex text: {prompt}\n\nSimple version:"
            max_tokens = 1000
        elif function_type == "creative":
            if prompt.startswith("song:"):
                topic = prompt[5:].strip()
                formatted_prompt = f"Write song lyrics about: {topic}\n\nLyrics:\n[Verse 1]"
            elif prompt.startswith("poem:"):
                topic = prompt[5:].strip()
                formatted_prompt = f"Write a poem about: {topic}\n\nPoem:"
            elif prompt.startswith("essay:"):
                topic = prompt[6:].strip()
                formatted_prompt = f"Write an essay about: {topic}\n\nEssay:"
            elif prompt.startswith("dialogue:"):
                topic = prompt[9:].strip()
                formatted_prompt = f"Write a dialogue about: {topic}\n\nDialogue:"
            elif prompt.startswith("script:"):
                topic = prompt[7:].strip()
                formatted_prompt = f"Write a script about: {topic}\n\nScript:"
            elif prompt.startswith("blog:"):
                topic = prompt[5:].strip()
                formatted_prompt = f"Write a blog post about: {topic}\n\nBlog Post:"
            else:
                formatted_prompt = f"Write a complete creative story about: {prompt}\n\nStory:"
            max_tokens = None  # No limit - let model decide when to stop
        else:
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            max_tokens = 2000
        
        # Handle simple greetings and short inputs before model generation
        if prompt.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']:
            return "Hello! I'm SRUSTI, your AI assistant. I'm here to help you with questions, creative writing, summaries, and more. What can I do for you today?"
        
        if len(prompt.strip()) < 3:
            return "I'd love to help! Could you please provide a bit more detail about what you'd like me to assist you with?"
        
        try:
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1800)
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise ValueError("Failed to process input text")
        
        with torch.no_grad():
            try:
                # No token limits - generate until natural end for all types
                output_sequences = model.generate(
                    **inputs,
                    temperature=0.8 if function_type == "creative" else 0.5,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError("Insufficient memory for generation. Try a shorter prompt.")
                raise
        
        full_response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        full_response = full_response.replace(formatted_prompt, "").strip()
        
        # Clean up response - remove common stop patterns and repetition
        stop_patterns = ["\nQuestion:", "\nText:", "\nOriginal:", "\nComplex text:", "\nWrite"]
        for pattern in stop_patterns:
            if pattern in full_response:
                full_response = full_response.split(pattern)[0].strip()
        
        # Allow complete stories - minimal processing
        
        # Remove leading dashes or bullets
        full_response = full_response.lstrip('- •*')
        
        if not full_response or len(full_response) < 5:
            # Generate appropriate responses for different inputs
            if prompt.lower().strip() in ['hi', 'hello', 'hey', 'greetings']:
                full_response = "Hello! I'm SRUSTI, your AI assistant. How can I help you today?"
            elif function_type == "question":
                full_response = f"I'd be happy to help answer your question about '{prompt[:50]}...' Please feel free to ask me anything!"
            elif function_type == "summary":
                full_response = f"I can help summarize the text you provided. The main topic appears to be about {prompt[:30]}..."
            elif function_type == "creative":
                full_response = f"Here's a creative piece about {prompt}: Let me create something interesting for you..."
            else:
                full_response = f"I understand you're asking about '{prompt[:50]}...' I'm here to help with any questions or tasks you have!"
        
        full_response = sanitize_response(full_response)
        
        return full_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to generate response: {str(e)}")



def paraphrase_text(text):
    """Paraphrase text using TinyLlama"""
    if len(text) > 500:
        text = text[:500] + "..."
    return get_tinyllama_response(text, "paraphrase")

def analyze_sentiment(text):
    """Analyze sentiment using VADER and TextBlob"""
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("Please provide text to analyze.")
        
        # Limit text length for processing
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        # VADER analysis (better for social media text)
        vader_scores = sia.polarity_scores(text)
        
        # TextBlob analysis (good for general text)
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Determine overall sentiment from VADER
        compound = vader_scores['compound']
        if compound >= 0.05:
            sentiment = "Positive"
            emoji = "😊"
        elif compound <= -0.05:
            sentiment = "Negative"
            emoji = "😞"
        else:
            sentiment = "Neutral"
            emoji = "😐"
        
        # Determine confidence
        confidence = abs(compound)
        if confidence >= 0.7:
            confidence_level = "Very High"
        elif confidence >= 0.4:
            confidence_level = "High"
        elif confidence >= 0.1:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        analysis = f"""**SENTIMENT ANALYSIS RESULTS:**

{emoji} **Overall Sentiment:** {sentiment}
🎯 **Confidence Level:** {confidence_level}

**VADER Scores:**
• Positive: {vader_scores['pos']:.2f}
• Neutral: {vader_scores['neu']:.2f}
• Negative: {vader_scores['neg']:.2f}
• Compound: {compound:.2f}

**TextBlob Scores:**
• Polarity: {textblob_polarity:.2f} (-1 to +1)
• Subjectivity: {textblob_subjectivity:.2f} (0 to 1)

**Interpretation:**
• VADER is optimized for social media text
• TextBlob provides general sentiment analysis
• Compound score > 0.05 = Positive, < -0.05 = Negative
• Higher absolute values indicate stronger sentiment"""
        
        return analysis
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise RuntimeError("Sentiment analysis failed. Please try again with different text.")

# Load feedback on startup
load_feedback()

# Start model loading in background thread
model_thread = threading.Thread(target=load_model_background, daemon=True)
model_thread.start()

@app.route('/')
def index():
    """Render the main page"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    return render_template('index.html')

@app.route('/model_status')
def model_status():
    """Return the current status of model loading"""
    return jsonify(model_loading_status)



@app.route('/generate', methods=['POST'])
def generate():
    """Generate a response based on the input"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt', '').strip()
        function_type = data.get('function_type', 'question')
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt cannot be empty"}), 400
        
        if len(prompt) > 5000:
            return jsonify({"success": False, "error": "Prompt too long (max 5000 characters)"}), 400
        
        response = get_fast_response(prompt, function_type)
        
        # Save to history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
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
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500



@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Paraphrase text"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        if len(text) > 2000:
            return jsonify({"success": False, "error": "Text too long (max 2000 characters)"}), 400
        
        paraphrased = get_fast_response(text, "paraphrase")
        
        # Save to history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        session['conversation_history'].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "paraphrase",
            "prompt": text,
            "response": paraphrased
        })
        session.modified = True
        
        return jsonify({
            "success": True,
            "original": text,
            "paraphrased": paraphrased
        })
    except Exception as e:
        logger.error(f"Error in paraphrase: {e}")
        return jsonify({"success": False, "error": "Failed to paraphrase text"}), 500

@app.route('/sentiment', methods=['POST'])
def sentiment():
    """Analyze sentiment of text"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        if len(text) > 5000:
            return jsonify({"success": False, "error": "Text too long (max 5000 characters)"}), 400
        
        analysis = analyze_sentiment(text)
        
        # Save to history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        session['conversation_history'].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "sentiment",
            "prompt": text,
            "response": analysis
        })
        session.modified = True
        
        return jsonify({
            "success": True,
            "text": text,
            "analysis": analysis
        })
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return jsonify({"success": False, "error": "Failed to analyze sentiment"}), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """Export content as PDF"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        title = data.get('title', 'SRUSTI Export').strip()
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({"success": False, "error": "No content provided"}), 400
        
        if len(content) > 50000:
            return jsonify({"success": False, "error": "Content too long for PDF export"}), 400
        
        # Use temporary directory instead of UPLOAD_FOLDER
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = f"srusti_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(temp_dir, filename)
            
            try:
                doc = SimpleDocTemplate(filepath, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor='#667eea',
                    spaceAfter=30,
                    alignment=1
                )
                
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 0.3*inch))
                
                body_style = ParagraphStyle(
                    'CustomBody',
                    parent=styles['BodyText'],
                    fontSize=11,
                    leading=14,
                    alignment=4
                )
                
                for para in content.split('\n'):
                    if para.strip():
                        story.append(Paragraph(para, body_style))
                        story.append(Spacer(1, 0.1*inch))
                
                doc.build(story)
                
                with open(filepath, 'rb') as f:
                    pdf_data = f.read()
                
                return Response(
                    pdf_data,
                    mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment;filename={filename}'}
                )
            except Exception as e:
                logger.error(f"PDF generation error: {e}")
                return jsonify({"success": False, "error": "Failed to generate PDF"}), 500
                
    except Exception as e:
        logger.error(f"Error in export_pdf: {e}")
        return jsonify({"success": False, "error": "PDF export failed"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Collect feedback on a response"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        function_type = data.get('function_type', '')
        prompt = data.get('prompt', '')
        response = data.get('response', '')
        rating = data.get('rating', 0)
        comments = data.get('comments', '')
        
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            return jsonify({"success": False, "error": "Rating must be between 1 and 5"}), 400
        
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "function_type": function_type,
            "prompt": prompt[:500],  # Limit prompt length
            "response": response[:1000],  # Limit response length
            "rating": rating,
            "comments": comments[:500]  # Limit comments length
        }
        
        feedback_data.append(feedback_entry)
        save_feedback()
        
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!"
        })
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({"success": False, "error": "Failed to save feedback"}), 500

@app.route('/history', methods=['GET'])
def history():
    """Get conversation history from the session"""
    try:
        history_data = session.get('conversation_history', [])
        return jsonify({
            "success": True,
            "history": history_data
        })
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({
            "success": True,
            "history": []
        })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the conversation history from the session"""
    try:
        session['conversation_history'] = []
        session.modified = True
        return jsonify({
            "success": True,
            "message": "Conversation history cleared"
        })
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to clear history"
        }), 500

@app.route('/download_history', methods=['GET'])
def download_history():
    """Generate a downloadable JSON file of the conversation history"""
    try:
        history_data = session.get('conversation_history', [])
        if not history_data:
            history_data = [{
                "message": "No conversation history available", 
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
        
        history_json = json.dumps(history_data, indent=2, ensure_ascii=False)
        filename = f'conversation_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        return Response(
            history_json,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment;filename={filename}'
            }
        )
    except Exception as e:
        logger.error(f"Download history error: {e}")
        return jsonify({"success": False, "error": "Failed to download history"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": model is not None,
        "model_status": model_loading_status
    })

@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors"""
    return jsonify({
        "success": False,
        "error": "Bad request - invalid input data"
    }), 400

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "The requested resource was not found"
    }), 404

@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large errors"""
    return jsonify({
        "success": False,
        "error": "File too large - maximum size is 16MB"
    }), 413

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        "success": False,
        "error": "An internal server error occurred"
    }), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    if not os.path.exists('templates/index.html'):
        logger.warning("templates/index.html not found.")
    
    logger.info("Starting SRUSTI AI Assistant with TinyLlama Backend...")
    logger.info("Access the web interface at http://localhost:5000")
    logger.info(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    logger.info("Model loading in background... this may take 2-5 minutes on first run")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
