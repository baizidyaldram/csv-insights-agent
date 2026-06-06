import os
import requests
import streamlit as st

# Correct OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Using gpt-oss-120b model
FREE_MODELS = [
    "gryphe/mythomax-l2-13b:free",  # Fallback 1
    "gpt-oss-120b",  # Your desired model
    "mistralai/mistral-7b-instruct:free",  # Fallback 2
]

# Primary model to use
PRIMARY_MODEL = "gpt-oss-120b"


def get_api_key() -> str | None:
    """Get API key from env or Streamlit secrets."""
    # First try environment variable
    key = os.getenv("OPENROUTER_API_KEY")
    
    # Then try Streamlit secrets
    if not key:
        try:
            key = st.secrets["OPENROUTER_API_KEY"]
            print("✅ API key loaded from Streamlit secrets")
        except Exception as e:
            print(f"❌ Could not load from secrets: {e}")
    
    return key


def call_llm(prompt: str, system: str = "", model_index: int = 0, max_tokens: int = 1500) -> str:
    """
    Call OpenRouter with gpt-oss-120b model.
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

**Your current secret:** OPENROUTER_API_KEY is set in Streamlit secrets but not being read properly."""

    # Use gpt-oss-120b as primary model
    if model_index == 0:
        model = PRIMARY_MODEL
    else:
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
        print(f"📤 Sending request to OpenRouter with model: {model}")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
        
        print(f"📥 Response status: {response.status_code}")
        
        # Handle different error codes
        if response.status_code == 401:
            return "❌ **Invalid API Key**\n\nYour OpenRouter API key appears to be invalid. Please check your Streamlit secrets."
        elif response.status_code == 402:
            # Try fallback model if primary requires payment
            if model_index == 0 and len(FREE_MODELS) > 1:
                print("🔄 Primary model requires payment, trying fallback...")
                return call_llm(prompt, system, model_index + 1, max_tokens)
            return "❌ **Insufficient Credits**\n\nThe model requires credits. Try a different model or add credits to your OpenRouter account."
        elif response.status_code == 404:
            return f"""❌ **Model Not Found (404)**

The model '{model}' might not be available or the name is incorrect.

**Try these alternatives:**
- gryphe/mythomax-l2-13b:free
- mistralai/mistral-7b-instruct:free
- meta-llama/llama-3.2-3b-instruct:free

**What would you like to do?**
1. Check available models at https://openrouter.ai/models
2. Use a different model by updating PRIMARY_MODEL in llm.py
3. Continue with fallback models"""

        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()
        print(f"✅ Successfully got response (length: {len(result)} chars)")
        return result
        
    except requests.exceptions.Timeout:
        return "⏰ **Request Timeout**\n\nThe API request took too long. Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection Error**\n\nCould not connect to OpenRouter API. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        # Try fallback model on error
        if model_index + 1 < len(FREE_MODELS):
            print(f"🔄 Trying fallback model...")
            return call_llm(prompt, system, model_index + 1, max_tokens)
        return f"❌ **API Error**\n\n{str(e)}"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return f"❌ **Unexpected Error**\n\n{str(e)}"


def get_active_model(index: int = 0) -> str:
    """Return the active model name"""
    if index == 0:
        return PRIMARY_MODEL
    return FREE_MODELS[index % len(FREE_MODELS)]


def test_api_connection() -> dict:
    """Test if the API key is working and model is accessible"""
    api_key = get_api_key()
    if not api_key:
        return {"success": False, "message": "No API key found"}
    
    # Test with a simple request
    test_prompt = "Say 'API working' if you can read this."
    
    try:
        response = call_llm(test_prompt, max_tokens=50)
        if "API working" in response or len(response) > 0:
            return {"success": True, "message": f"API is working! Model: {PRIMARY_MODEL}"}
        else:
            return {"success": False, "message": "API returned unexpected response"}
    except Exception as e:
        return {"success": False, "message": f"API test failed: {e}"}
