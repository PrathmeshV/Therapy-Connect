import os
import time
import json
import shutil
from emotion_service import EmotionClassifier
from rag_service import RAG_LLM_System

# --- Configuration ---
BASE_IO_DIR = "../pipeline_io/"
TRANSCRIPTION_INPUT_DIR = os.path.join(BASE_IO_DIR, "stage1_transcriptions")
CLASSIFIED_OUTPUT_DIR = os.path.join(BASE_IO_DIR, "stage2_classified")

def main():
    print("--- 🚀 Starting Pipeline Orchestrator ---")
    
    # Initialize services
    emotion_classifier = EmotionClassifier()
    rag_llm_system = RAG_LLM_System()

    # Create I/O directories if they don't exist
    os.makedirs(os.path.join(TRANSCRIPTION_INPUT_DIR, "archive"), exist_ok=True)
    os.makedirs(os.path.join(CLASSIFIED_OUTPUT_DIR, "archive"), exist_ok=True)
    
    print(f"\n👀 Watching for new files in: {os.path.abspath(TRANSCRIPTION_INPUT_DIR)}")

    while True:
        try:
            # Look for new transcriptions from the Whisper service
            files = [f for f in os.listdir(TRANSCRIPTION_INPUT_DIR) if f.endswith('.txt')]

            if not files:
                time.sleep(2) # Wait 2 seconds before checking again
                continue

            filename = files[0] # Process one file at a time
            input_filepath = os.path.join(TRANSCRIPTION_INPUT_DIR, filename)
            
            print(f"\n[▶️] New file detected: {filename}")

            # --- Stage 1: Read Transcription ---
            with open(input_filepath, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            if not original_text.strip():
                print("[WARN] File is empty. Archiving and skipping.")
                shutil.move(input_filepath, os.path.join(TRANSCRIPTION_INPUT_DIR, "archive", filename))
                continue

            # --- Stage 2: Emotion Classification ---
            print("    -> Classifying emotion...")
            emotion_data = emotion_classifier.predict(original_text)
            
            # --- Stage 3: RAG + LLM Generation ---
            print("    -> Generating RAG response...")
            final_response = rag_llm_system.generate_final_response(original_text, emotion_data)
            
            print("\n" + "="*40)
            print("✅✅✅ FINAL RESPONSE ✅✅✅")
            print(final_response)
            print("="*40 + "\n")

            # --- Stage 4: Cleanup ---
            shutil.move(input_filepath, os.path.join(TRANSCRIPTION_INPUT_DIR, "archive", filename))
            print(f"[*] Processed and archived {filename}.")
            print(f"👀 Watching for new files...")

        except Exception as e:
            print(f"[‼️] An error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()