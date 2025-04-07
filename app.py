from flask import Flask, render_template, request, jsonify
import os
import threading
import time
from dotenv import load_dotenv
import logging

# Import your existing Galatea components
from import_random import GalateaAI, DialogueEngine, AvatarEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables to hold components
galatea_ai = None
dialogue_engine = None
avatar_engine = None
is_initialized = False
initializing = False

def initialize_components():
    """Initialize Galatea components in a separate thread"""
    global galatea_ai, dialogue_engine, avatar_engine, is_initialized, initializing
    
    initializing = True
    logging.info("Starting to initialize Galatea components...")
    
    try:
        # Initialize components
        galatea_ai = GalateaAI()
        dialogue_engine = DialogueEngine(galatea_ai)
        avatar_engine = AvatarEngine()
        avatar_engine.update_avatar(galatea_ai.emotional_state)
        
        is_initialized = True
        logging.info("Galatea components initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Galatea: {e}")
    finally:
        initializing = False

# Start initialization in a separate thread
init_thread = threading.Thread(target=initialize_components)
init_thread.daemon = True
init_thread.start()

@app.route('/')
def home():
    # Add error handling for template rendering
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        return f"Error loading the application: {e}. Make sure templates/index.html exists.", 500

@app.route('/api/chat', methods=['POST'])
def chat():
    # Check if components are initialized
    if not is_initialized:
        return jsonify({
            'response': 'I am still initializing. Please try again in a moment.',
            'avatar_shape': 'Circle',
            'emotions': {'joy': 0.2, 'sadness': 0.2, 'anger': 0.2, 'fear': 0.2, 'curiosity': 0.2},
            'is_initialized': False
        })
    
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Process the message through Galatea
        response = dialogue_engine.get_response(user_input)
        
        # Update avatar
        avatar_engine.update_avatar(galatea_ai.emotional_state)
        avatar_shape = avatar_engine.avatar_model
        
        # Get emotional state for frontend
        emotions = {k: round(v, 2) for k, v in galatea_ai.emotional_state.items()}
        
        return jsonify({
            'response': response,
            'avatar_shape': avatar_shape,
            'emotions': emotions,
            'is_initialized': True
        })
    except Exception as e:
        logging.error(f"Error processing chat: {e}")
        return jsonify({
            'error': 'Failed to process your message',
            'details': str(e)
        }), 500

@app.route('/health')
def health():
    """Simple health check endpoint to verify the server is running"""
    return jsonify({
        'status': 'ok',
        'is_initialized': is_initialized,
        'initializing': initializing,
        'gemini_available': galatea_ai.gemini_available if galatea_ai else False
    })

@app.route('/status')
def status():
    """Status endpoint to check initialization progress"""
    return jsonify({
        'is_initialized': is_initialized,
        'initializing': initializing
    })

if __name__ == '__main__':
    print("Starting Galatea Web Interface...")
    print("The chatbot will continue loading in the background.")
    print("Open your browser and navigate to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)  # Set debug=True for development, False for production
