import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def check_available_models():
    """Check and print all available Gemini models"""
    try:
        # Get API key from environment variable 
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("Gemini API key not found in environment variables.")
            api_key = input("Enter your Gemini API key: ")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # List available models
        print("Fetching available models...")
        models = genai.list_models()
        
        print("\n===== AVAILABLE GOOGLE AI MODELS =====")
        for model in models:
            print(f"- {model.name}")
            
        print("\n===== RECOMMENDED MODELS TO USE =====")
        for model in models:
            if "gemini-1.5" in model.name:
                print(f"✓ {model.name}")
                
        return [model.name for model in models]
    
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

if __name__ == "__main__":
    check_available_models()
    print("\nYou can use any of these models in your application.")
    print("To fix your application, update the model name in initialize_gemini() method.")
    print("Example usage: self.gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')")
