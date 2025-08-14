# Digital Galatea AI

Digital Galatea is a conversational AI with a dynamic emotional model. It features a web-based interface where an avatar's shape and expression change in real-time to reflect the AI's feelings, which are influenced by the conversation.

## Features

- **Conversational AI**: Powered by the Google Gemini API for natural and engaging conversations.
- **Dynamic Emotional Model**: Simulates five core emotions: Joy, Sadness, Anger, Fear, and Curiosity.
- **Responsive Avatar**: The AI's visual avatar changes its shape and facial expression based on its dominant emotion.
- **Sentiment Analysis**: Analyzes user input to dynamically update the AI's emotional state. It uses Azure Text Analytics for high accuracy when configured, with a seamless fallback to a local NLTK VADER model.
- **Real-time Web Interface**: Built with Flask and JavaScript, the interface polls for updates to keep the avatar and emotion bars synchronized with the AI's state.

## Tech Stack

- **Backend**: Python, Flask
- **AI & Machine Learning**:
  - Google Gemini API
  - Azure Cognitive Service for Language (Text Analytics)
  - NLTK (VADER)
- **Frontend**: HTML, CSS, JavaScript
- **Environment Management**: `python-dotenv`

## Project Structure

```
.
├── app.py                  # Main Flask application, API endpoints
├── import_random.py        # Core AI logic (GalateaAI, DialogueEngine, AvatarEngine)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── static/
│   ├── css/style.css       # Styles for the web interface
│   └── js/script.js        # Frontend JavaScript for interactivity
└── templates/
    └── index.html          # Main HTML file for the web interface
```

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd Digital-Galatea
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    - Create a file named `.env` in the project root.
    - Add your Google Gemini API key to it. You can get a key from the [Google AI Studio](https://ai.google.dev/).

    ```properties
    # .env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

5.  **(Optional) Configure Azure Text Analytics**
    - For more accurate sentiment analysis, you can use Azure.
    - Get your key and endpoint from the Azure Portal after creating a "Language service" resource.
    - Add them to your `.env` file:
    ```properties
    # .env
    AZURE_TEXT_ANALYTICS_KEY=your_azure_key_here
    AZURE_TEXT_ANALYTICS_ENDPOINT=your_azure_endpoint_here
    ```
    If these are not provided, the application will automatically use the built-in NLTK sentiment analyzer.

## How to Run

1.  **Start the Flask Application**
    ```bash
    python app.py
    ```

2.  **Access the Web Interface**
    - Open your web browser and navigate to `http://127.0.0.1:5000`.
    - The AI will initialize in the background. Once ready, you can start chatting.

## API Endpoints

The application exposes the following endpoints:

-   `GET /`: Serves the main chat interface.
-   `POST /api/chat`: Handles chat messages and returns the AI's response.
-   `GET /api/avatar`: Provides the current avatar shape, emotions, and sentiment for real-time frontend updates.
-   `GET /status`: Reports the initialization status of the AI components.
-   `GET /health`: A simple health check endpoint.
