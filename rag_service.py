# services/rag_service.py
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "../data/knowledge_base_db"
COLLECTION_NAME = "therapeutic_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = 'gemini-2.5-pro'

class RAG_LLM_System:
    def __init__(self):
        print("🤖 Initializing RAG+LLM System...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        
        # This will now find the key loaded from your .env file
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Make sure it is set in your .env file.")
            
        genai.configure(api_key=api_key)
        self.llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
        print("✅ RAG+LLM System is ready.")

    def _search_knowledge_base(self, query_text: str, n_results=10):
        query_embedding = self.embedding_model.encode(query_text).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results['documents'][0]

    def generate_final_response(self, original_text: str, emotion_data: dict):
        emotion = emotion_data.get('emotion', 'neutral')
        search_query = f"Therapeutic advice for someone feeling {emotion}."
        retrieved_context = self._search_knowledge_base(search_query)
        context_str = "\n\n---\n\n".join(retrieved_context)
        prompt = f"""You are an empathetic therapeutic assistant. A user shared this: '{original_text}'. They are feeling {emotion}. Based on this, and the knowledge that '{context_str}', provide multiple, supportive paragraph (400-500 words) that acknowledges their feelings, integrates a key insight, and offers one gentle, actionable suggestion."""
        response = self.llm_model.generate_content(prompt)
        return response.text.strip()