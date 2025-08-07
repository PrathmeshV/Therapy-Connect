# services/whisper.py
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import whisper

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Whisper model...")
    ml_models["whisper"] = whisper.load_model("base")
    print("Whisper model loaded.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/submit-transcription")
async def submit_transcription(file: UploadFile = File(...)):
    # Define the output directory
    TRANSCRIPTION_OUTPUT_DIR = "../pipeline_io/stage1_transcriptions"
    os.makedirs(TRANSCRIPTION_OUTPUT_DIR, exist_ok=True)
    
    # Transcribe audio
    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = file.filename
    result = ml_models["whisper"].transcribe(audio_buffer, fp16=False)
    
    # Save transcription to a file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"transcription_{timestamp}.txt"
    output_path = os.path.join(TRANSCRIPTION_OUTPUT_DIR, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
        
    return {"message": "File transcribed successfully.", "filepath": output_path}