# Tech Stack — BolKeBatao AI

BolKeBatao AI is an AI-powered, voice-first clinical history intake and structuring platform designed to transcribe multilingual patient speech (Hindi, Hinglish, English) and extract structured clinical data with verifiable evidence.

---

## 1. Core Architecture & Language
- **Language**: Python 3.10+ / 3.13
- **Paradigm**: Event-driven Web App with Real-time Media Streaming & Asynchronous Processing

---

## 2. Frontend & User Interface
- **Streamlit (`v1.40+`)**: Interactive clinical web dashboard and physician review workstation.
- **Custom CSS / Glassmorphic UI**: High-contrast, responsive clinical theme designed for low physician fatigue.
- **HTML5 Audio / Browser Media API**: Client-side audio recording and playback integration.

---

## 3. Real-Time Audio Capture & Processing
- **Streamlit-WebRTC (`streamlit-webrtc >= 0.47`)**: Real-time browser microphone streaming over WebRTC peer connection.
- **PyAV (`av >= 12`)**: High-performance audio frame decoding, demuxing, and resampling.
- **NumPy (`numpy >= 1.26`)**: In-memory PCM audio buffer management and mono/stereo 16-bit array transformation.
- **Wave / BytesIO**: In-memory WAV packaging and payload serialization.
- **VAD (Voice Activity Detection)**: Energy-based silence and speech activity detection for automatic audio capture.

---

## 4. Artificial Intelligence & LLMs
- **Google GenAI SDK (`google-genai >= 2.0.0`)**: Official modern Google Gemini SDK.
- **Gemini 2.5 Flash / Gemini 1.5 Pro**:
  - **Speech-to-Text (Audio Modality)**: Native multimodal audio understanding and verbatim multilingual transcription (Hindi, Hinglish, English, Code-mixed).
  - **Clinical Information Extraction**: Zero-shot structured JSON extraction grounded with verbatim patient quotes.
  - **Verification & Guardrails**: Hallucination check against source audio transcripts to prevent ungrounded clinical claims.

---

## 5. Security & Secret Management
- **Streamlit Secrets (`.streamlit/secrets.toml`)**: Safe server-side credential storage.
- **Environment Fallbacks**: Unified key resolution across `GEMINI_API_KEY` and `GOOGLE_API_KEY`.
- **Git Shielding**: `.gitignore` configured to ensure no sensitive medical data or API credentials are leaked.

---

## 6. Supported Audio Formats
- **WAV (`audio/wav`)**
- **MP3 (`audio/mp3`)**
- **M4A / AAC (`audio/m4a`)**
- **WebM (`audio/webm`)**
- **Raw PCM 16-bit Linear Audio**

---

## 7. Dependencies Summary (`requirements.txt`)
```text
streamlit>=1.40
streamlit-webrtc>=0.47
google-genai>=2.0.0
av>=12
numpy>=1.26
```
