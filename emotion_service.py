# services/emotion_service.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "../models/BiLSTM_model.h5"
TOKENIZER_PATH = "../models/final_tokenizer.pkl"
MAX_LEN = 100

class EmotionClassifier:
    def __init__(self):
        print("🧠 Loading emotion classification model...")
        self.model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.labels = ['Sadness', 'joy', 'love', 'angry', 'fear', 'surprise'] # labels
        print("✅ Emotion classifier loaded.")

    def _clean_text(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

    def predict(self, text: str):
        cleaned_text = self._clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
        probabilities = self.model.predict(padded_sequence)[0]
        top_emotion_index = np.argmax(probabilities)

        # Create a dictionary of all emotion scores
        all_scores = {self.labels[i]: round(float(score), 4) for i, score in enumerate(probabilities)}

        # Return the top emotion plus the dictionary of all scores
        return {
            "emotion": self.labels[top_emotion_index],
            "confidence_score": round(float(probabilities[top_emotion_index]), 4),
            "all_scores": all_scores
        }