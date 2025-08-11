# Therapy Connect 🧠✨

Therapy Connect is an AI-powered interactive therapeutic assistant designed to provide an empathetic and understanding space for users to explore their thoughts. It's a full-stack web application featuring a multi-stage AI pipeline that listens, understands, and interacts in a deeply personalized and conversational manner.

# 🏛️ High-Level Architecture

The application operates on a client-server model, with a dynamic web interface communicating with a unified Python backend that houses the entire AI pipeline.



# 🌊 Data Flow Diagram

The core logic resides within the FastAPI server. Here is a detailed breakdown of how data flows through the AI services during a user session.

![Therapy Connect Data Flow](./therapy-connect-flow.png)


# ✨ Key Features
- **End-to-End AI Pipeline**: A complete system processing raw audio through transcription, emotion analysis, and context-aware response generation.

- **Intelligent RAG System**: Utilizes Retrieval-Augmented Generation with a ChromaDB vector database to ground the LLM's responses in a specialized knowledge base, ensuring accuracy and relevance.

- **Dynamic Conversational AI**: Engages users in an interactive dialogue by generating clarifying questions based on their initial statement and emotional state.

- **Nuanced Emotion Analysis**: A custom BiLSTM model identifies both primary and secondary emotions for a deeper understanding of the user's state.

- **Interactive Web Interface**: A modern, responsive frontend built with HTML, Tailwind CSS, and JavaScript, featuring real-time audio visualization and dynamic state management.

- **Downloadable Summaries**: Users can download a complete summary of their session, including their responses and the AI's analysis, as a formatted PDF.

- **Extensible Knowledge Base**: Includes a utility script (ingest_document.py) to automatically process and add new knowledge from PDF documents into the vector database.

# 📂 Project Structure

<pre>
my_ai_pipe/
├── components/
│   └── UI/
│       └── chart.js
│
├── data/
│   └── knowledge_base_db/
│
├── dev files/
│   ├── asset/
│   │   ├── CSS/
│   │   │   └── style.css
│   │   └── JavaScript/
│   │       ├── analytics.js
│   │       ├── app.js
│   │       ├── history.js
│   │       └── settings.js
│   ├── template/
│   │   └── (empty)
│   ├── analytics.html
│   ├── history.html
│   ├── index.html
│   └── setting.html
│
├── models/
│   ├── BiLSTM_model.h5
│   └── final_tokenizer.pkl
│
├── pipeline_io/
│   └── audio_uploads/
│
├── services/
│   ├── main_app.py
│   ├── emotion_service.py
│   └── rag_service.py
│
├── .env
└── requirements.txt
</pre>


# 🚀 Setup and Installation

Follow these steps to set up and run the project locally.

## Prerequisites

- Python 3.9+

- FFmpeg: Whisper requires FFmpeg to be installed on your system and accessible from the command line. You can download it from ffmpeg.org.

- Google API Key: You need a Google API key with the "Generative Language API" enabled to use Gemini Pro.

## Installation Steps

- Clone the repository:

  - git clone [your-repository-url]
  - cd My_AI_Pipeline


- Create a virtual environment:

  - python -m venv venv
  - source venv/bin/activate  # On Windows: venv\Scripts\activate


- Install dependencies:

  - pip install -r requirements.txt


- Set up your environment variables:

  - Create a file named .env in the root directory of the project.

  - Add your Google API key to this file:

    - GOOGLE_API_KEY="your_google_api_key_here"


- Build the Knowledge Base (Optional but Recommended):

  - You can expand the knowledge base by adding your own PDF documents. Place the ingest_document.py script in the root directory.

  - Run the script from your terminal, providing the path to a PDF:

    - python ingest_document.py "path/to/your/document.pdf"
  

# ▶️ How to Run
1. Start the Backend Server:

  - Navigate to the services/ directory.

  - Run the following command in your terminal:

    - uvicorn main_app:app --reload

  - The server will start on http://127.0.0.1:8000.


2. Open the Frontend:

  - Navigate to the root directory of the project.
  
  - Open the index.html file in your web browser.

The application is now ready to use!

# 🔮 Future Scope

- **User Authentication**: Implement a user login system to save and track session history securely.

- **Analytics Dashboard**: Develop a dashboard to visualize emotional trends and session data over time.

- **Streaming Audio**: Upgrade from file uploads to real-time streaming audio transcription for a more fluid user experience.

- **Cloud Deployment**: Deploy the entire application to a cloud platform like Google Cloud Run or AWS for public access.
