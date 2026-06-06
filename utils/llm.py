import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Updated list of working free models (as of 2024)
FREE_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

def get_api_key() -> str | None:
    """Get API key from env or Streamlit secrets."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
    return key

def call_llm(prompt: str, system: str = "", model_index: int = 0, max_tokens: int = 1500) -> str:
    """
    Call OpenRouter with fallback across free models.
    """
    api_key = get_api_key()
    if not api_key:
        return "⚠️ **API Key Missing**\n\nPlease add your OpenRouter API key to continue. Get one at https://openrouter.ai/keys"

    model = FREE_MODELS[model_index % len(FREE_MODELS)]
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://csv-insight-agents.streamlit.app",
        "X-Title": "CSV Insight Agents",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        
        # Check for specific error codes
        if response.status_code == 401:
            return "❌ **Invalid API Key**\n\nYour OpenRouter API key is invalid. Please check your credentials."
        elif response.status_code == 402:
            return "❌ **Insufficient Credits**\n\nYour OpenRouter account has insufficient credits for this request."
        elif response.status_code == 404:
            return "❌ **API Endpoint Error**\n\nThe OpenRouter API endpoint could not be reached. This might be a temporary issue. Please try again later or contact OpenRouter support."
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.Timeout:
        return "⏰ **Request Timeout**\n\nThe request took too long. Please try again."
    except requests.exceptions.RequestException as e:
        # Try next model on failure
        if model_index + 1 < len(FREE_MODELS):
            return call_llm(prompt, system, model_index + 1, max_tokens)
        return f"❌ **API Error**\n\n{str(e)}"
    except Exception as e:
        return f"❌ **Unexpected Error**\n\n{str(e)}"

def get_active_model(index: int = 0) -> str:
    return FREE_MODELS[index % len(FREE_MODELS)]
