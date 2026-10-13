import os
import streamlit as st


def get_api_key():
    """Read Gemini API key from Streamlit secrets or environment variables.

    Supports GEMINI_API_KEY, GOOGLE_API_KEY, and backward-compatible OPENAI_API_KEY
    if configured in .streamlit/secrets.toml or the environment.
    """
    key_names = [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_KEY",
        "OPENAI_API_KEY",  # Supported for seamless backward compatibility
    ]

    for name in key_names:
        try:
            val = st.secrets.get(name, "")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass

    for name in key_names:
        val = os.getenv(name, "")
        if val and val.strip():
            return val.strip()

    return ""
