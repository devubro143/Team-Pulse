# BolKeBatao AI — Ready Project

Voice-first multilingual clinical history assistant powered by Google Gemini.

## Folder layout
- `app.py`: Streamlit application with live microphone (WebRTC & native mic), Gemini transcription, and structured clinical extraction.
- `api_key.py`: Server-side API key loader (reads `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or existing key alias).
- `.streamlit/secrets.toml`: Server configuration containing your Gemini API key.
- `requirements.txt`: Python dependencies (`google-genai`, `streamlit`, `streamlit-webrtc`, etc.).

## Setup & Running on Windows
Open PowerShell in this folder and run:
```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m streamlit run app.py
```

## Gemini API Configuration
In `.streamlit/secrets.toml`, add your Google Gemini API key:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
*(Also supports `GOOGLE_API_KEY` or existing keys configured in secrets or environment variables).*
Never commit `secrets.toml` to version control.

## Usage
1. **Live Microphone / Recording**: Click START (WebRTC with automatic silence detection), use the native browser audio input, or upload an audio file (`.wav`, `.mp3`, `.m4a`, etc.).
2. **Audio Transcription**: Google Gemini directly transcribes spoken Hindi, Hinglish, English, or mixed speech verbatim.
3. **Structured Extraction**: Gemini extracts structured clinical history fields (`complaint`, `duration`, `symptom_pattern`, `severity`, `medicines`, `allergies`) with strict verbatim evidence grounding.
4. **Clinician Review & Export**: Review each verified field, edit notes, and export the structured record to JSON.
