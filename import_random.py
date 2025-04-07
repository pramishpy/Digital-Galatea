import random
import nltk  # Import NLTK for basic NLP tasks
import os
from dotenv import load_dotenv  # For loading environment variables from .env file
import google.generativeai as genai  # Add Google's Generative AI library
import logging  # Add logging for better error tracking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Download NLTK data (only needs to be done once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')
    
# Make sure punkt is downloaded before importing the rest
nltk.download('punkt', quiet=True)  # Add this to ensure the tokenizer is available

from transformers import pipeline  # Import Hugging Face Transformers
from enum import Enum #import

# --- 1. AI Core ---
class GalateaAI:
    def __init__(self):
        self.emotional_state = {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "curiosity": 0.0}
        self.knowledge_base = {}
        self.learning_rate = 0.05 # Reduced learning rate
        # Specify exact model name for sentiment analysis to avoid warnings
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                          model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
                                          revision="714eb0f")
        self.response_model = "A generic response" #Place Holder for the ML model
        
        # Initialize Gemini API
        self.initialize_gemini()
        
    def initialize_gemini(self):
        """Initialize the Gemini API with API key from .env file"""
        try:
            # Get API key from environment variable 
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                # Prompt user for API key if not found in environment variables
                logging.warning("Gemini API key not found in environment variables.")
                print("Gemini API key not found in environment variables.")
                print("Please create a .env file with your GEMINI_API_KEY or enter it now:")
                api_key = input("Enter your Gemini API key: ")
            
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # List available models to ensure we're using a valid one
            try:
                models = genai.list_models()
                available_models = [model.name for model in models]
                logging.info(f"Available Gemini models: {available_models}")
                
                # Look for newer models first - gemini-1.5-flash or gemini-1.5-pro
                preferred_models = [
                    "models/gemini-1.5-flash",
                    "models/gemini-1.5-pro", 
                    "models/gemini-1.5-flash-latest",
                    "models/gemini-1.5-pro-latest"
                ]
                
                # Find the first available preferred model
                model_name = None
                for preferred in preferred_models:
                    matching = [m for m in available_models if preferred in m]
                    if matching:
                        model_name = matching[0]
                        break
                
                # If no preferred model found, use any available model
                if not model_name:
                    model_name = available_models[0] if available_models else None
                    
                if not model_name:
                    raise ValueError("No suitable Gemini models available")
                    
                logging.info(f"Selected model: {model_name}")
                self.gemini_model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple prompt
                test_response = self.gemini_model.generate_content("Hello")
                logging.info("Test response received from Gemini API")
                self.gemini_available = True
                print(f"Gemini API initialized successfully with model: {model_name}")
                
            except Exception as e:
                logging.error(f"Error initializing Gemini model: {e}")
                raise
                
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {e}")
            print(f"Failed to initialize Gemini API: {e}")
            self.gemini_available = False
            # Don't completely fail - allow the system to run with fallback responses

    def process_input(self, user_input):
        sentiment_score = self.analyze_sentiment(user_input)
        keywords = self.extract_keywords(user_input)
        intent = self.determine_intent(user_input)

        # Enhanced Emotion Update (decay and normalization)
        for emotion in self.emotional_state:
            # Decay emotions (more realistic fading)
            self.emotional_state[emotion] *= 0.9  # Decay by 10% each turn
            # Normalize
            self.emotional_state[emotion] = max(0.0, min(1.0, self.emotional_state[emotion]))

        self.emotional_state["joy"] += sentiment_score * self.learning_rate
        self.emotional_state["sadness"] -= sentiment_score * self.learning_rate


        # Re-normalize
        total_emotion = sum(self.emotional_state.values())
        for emotion in self.emotional_state:
            self.emotional_state[emotion] /= total_emotion if total_emotion > 0 else 1

        self.update_knowledge(keywords, user_input)
        response = self.generate_response(intent, keywords, self.emotional_state, user_input)
        return response

    def analyze_sentiment(self, text):
        # Leverage Hugging Face's sentiment analysis pipeline
        result = self.sentiment_analyzer(text)[0]  # Get result
        sentiment = result['label']
        score = result['score']

        if sentiment == 'POSITIVE':
            return score
        else:
            return -score  # Negative sentiment

    def extract_keywords(self, text):
        # Attempt to download the specific resource mentioned in the error
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
        
        try:
            # Try using NLTK's tokenizer
            tokens = nltk.word_tokenize(text)
            keywords = [word.lower() for word in tokens if word.isalnum()]
            return keywords
        except LookupError:
            # Fall back to a simple split-based approach if NLTK fails
            # This is a simple alternative that doesn't require NLTK resources
            words = text.split()
            # Clean up words (remove punctuation)
            keywords = [word.lower().strip('.,!?;:()[]{}""\'') for word in words]
            # Filter out empty strings
            keywords = [word for word in keywords if word and word.isalnum()]
            return keywords

    def determine_intent(self, text):
        # More comprehensive intent recognition (using keywords)
        text = text.lower()
        if "what" in text or "how" in text or "why" in text:
            return "question"
        elif "thank" in text:
            return "gratitude"
        elif "goodbye" in text or "bye" in text:
            return "farewell"
        else:
            return "statement"

    def generate_response(self, intent, keywords, emotional_state, original_input):
        # Try to use Gemini API if available
        if hasattr(self, 'gemini_available') and self.gemini_available:
            try:
                # Create a prompt that includes emotional context and intent
                emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
                
                # Create a character prompt for Gemini
                prompt = f"""
                You are Galatea, an AI assistant with the following emotional state:
                {emotions_text}
                
                User input: "{original_input}"
                
                Respond in character as Galatea. Keep your response concise (under 50 words) and reflect your emotional state in your tone.
                If you're feeling more joy, be more enthusiastic. If sad, be more melancholic.
                """
                
                logging.info("Sending request to Gemini API")
                # Get response from Gemini with safety settings
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40
                }
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Check if response is valid and return it
                if response and hasattr(response, 'text'):
                    return response.text.strip()
                elif hasattr(response, 'parts'):
                    # Try alternate access method
                    return response.parts[0].text.strip()
                else:
                    logging.warning(f"Unexpected response format: {response}")
                    # Fall back to basic response
                    return "I'm processing that..."
            
            except Exception as e:
                logging.error(f"Error using Gemini API: {e}")
                print(f"Error using Gemini API: {e}")
                # Fall back to basic response logic
        
        # Original response generation logic as fallback
        if intent == "question":
            if "you" in keywords:
                return "I am, what you want me to be."
            else:
                return "I will have to look into that"
        elif intent == "gratitude":
            return "You're welcome!"
        else:
            return "Interesting."

    def update_knowledge(self, keywords, user_input):
      #for new key words remember them
        for keyword in keywords:
            if keyword not in self.knowledge_base:
                self.knowledge_base[keyword] = user_input

# --- 2. Dialogue Engine ---
class DialogueEngine:
    def __init__(self, ai_core):
        self.ai_core = ai_core

    def get_response(self, user_input):
        ai_response = self.ai_core.process_input(user_input)
        styled_response = self.apply_style(ai_response, self.ai_core.emotional_state)
        return styled_response

    def apply_style(self, text, emotional_state):
      style = self.get_style(emotional_state)
      #selects styles based on emotions
      #add style to text
      styled_text = text + f" ({style})"
      return styled_text

    def get_style(self, emotional_state):
        #determine style based on the state of the AI
        return "neutral"

# --- 3. Avatar Engine ---

class AvatarShape(Enum): #create shape types for the avatar
  CIRCLE = "Circle"
  TRIANGLE = "Triangle"
  SQUARE = "Square"

class AvatarEngine:
    def __init__(self):
        self.avatar_model = "Simple Circle"  # Start with a basic shape
        self.expression_parameters = {}

    def update_avatar(self, emotional_state):
        # Map emotions to avatar parameters (facial expressions, color)
        joy_level = emotional_state["joy"]
        sadness_level = emotional_state["sadness"]

        # Simple mapping (placeholder)
        self.avatar_model = self.change_avatar_shape(joy_level, sadness_level)

    def change_avatar_shape(self, joy, sad):
        #determine shape based on feelings
        if joy > 0.5:
            return AvatarShape.CIRCLE.value
        elif sad > 0.5:
            return AvatarShape.TRIANGLE.value
        else:
            return AvatarShape.SQUARE.value
            
    def render_avatar(self):
        # Simple console rendering of the avatar state
        print(f"Avatar shape: {self.avatar_model}")

# --- 4. Main Program Loop ---
# Download NLTK data again before starting the main loop to ensure availability
nltk.download('punkt', quiet=True)

try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  nltk.download('punkt')

#Create
galatea_ai = GalateaAI()
dialogue_engine = DialogueEngine(galatea_ai)
avatar_engine = AvatarEngine()
avatar_engine.update_avatar(galatea_ai.emotional_state)
# Initial avatar rendering
avatar_engine.render_avatar()

while True:
    user_input = input("You: ")
    response = dialogue_engine.get_response(user_input)
    print(f"Galatea: {response}")

    avatar_engine.update_avatar(galatea_ai.emotional_state)
    avatar_engine.render_avatar()