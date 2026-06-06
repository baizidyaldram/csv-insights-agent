import os
import requests
import streamlit as st

# Correct OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Working free models on OpenRouter (updated)
FREE_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free", 
    "microsoft/phi-3-mini-128k-instruct:free",
    "google/gemma-2-2b-it:free",
]


def get_api_key() -> str | None:
    """Get API key from env or Streamlit secrets."""
    # First try environment variable
    key = os.getenv("OPENROUTER_API_KEY")
    
    # Then try Streamlit secrets
    if not key:
        try:
            key = st.secrets["OPENROUTER_API_KEY"]
            print("✅ API key loaded from Streamlit secrets")  # Debug
        except Exception as e:
            print(f"❌ Could not load from secrets: {e}")
    
    if not key:
        print("⚠️ No API key found")
    else:
        print(f"🔑 API key found (length: {len(key)})")  # Debug
    
    return key


def call_llm(prompt: str, system: str = "", model_index: int = 0, max_tokens: int = 1500) -> str:
    """
    Call OpenRouter with fallback across free models.
    """
    api_key = get_api_key()
    if not api_key:
        return """⚠️ **OpenRouter API Key Not Found**

Please add your OpenRouter API key to continue.

**How to get a key:**
1. Go to https://openrouter.ai/keys
2. Sign up or log in
3. Create a new API key
4. Add it to Streamlit secrets

**For now, using simulated insights...**"""

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
        "temperature": 0.7,
    }

    try:
        print(f"📤 Sending request to OpenRouter with model: {model}")  # Debug
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        
        print(f"📥 Response status: {response.status_code}")  # Debug
        
        # Handle different error codes
        if response.status_code == 401:
            return "❌ **Invalid API Key**\n\nYour OpenRouter API key appears to be invalid. Please check your Streamlit secrets."
        elif response.status_code == 402:
            return "❌ **Insufficient Credits**\n\nYour OpenRouter account needs credits. Try a different free model or add credits."
        elif response.status_code == 404:
            return f"""❌ **API Endpoint Error (404)**

The OpenRouter API endpoint could not be reached. This might mean:

1. **The API URL has changed** - Please check OpenRouter documentation
2. **Temporary outage** - Try again in a few minutes
3. **Region restriction** - Your region might be blocked

**Debug Info:**
- URL: {OPENROUTER_API_URL}
- Model: {model}
- Status: {response.status_code}

**Workaround:** Using simulated insights for now."""

        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()
        print(f"✅ Successfully got response (length: {len(result)} chars)")  # Debug
        return result
        
    except requests.exceptions.Timeout:
        return "⏰ **Request Timeout**\n\nThe API request took too long. Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection Error**\n\nCould not connect to OpenRouter API. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")  # Debug
        # Try next model on failure
        if model_index + 1 < len(FREE_MODELS):
            print(f"🔄 Trying next model...")  # Debug
            return call_llm(prompt, system, model_index + 1, max_tokens)
        return f"❌ **API Error**\n\n{str(e)}"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")  # Debug
        return f"❌ **Unexpected Error**\n\n{str(e)}"


def get_active_model(index: int = 0) -> str:
    return FREE_MODELS[index % len(FREE_MODELS)]


# Test function to verify API key works
def test_api_connection() -> dict:
    """Test if the API key is working"""
    api_key = get_api_key()
    if not api_key:
        return {"success": False, "message": "No API key found"}
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "message": "API key is valid"}
        else:
            return {"success": False, "message": f"API key invalid (status {response.status_code})"}
    except Exception as e:
        return {"success": False, "message": f"Connection error: {e}"}
