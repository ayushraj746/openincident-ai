import os
import time
import requests
from dotenv import load_dotenv

# ---------------- LOAD ENV ---------------- #
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found. Please set it in .env file")

# ---------------- MODEL CONFIG ---------------- #
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# ---------------- MAIN FUNCTION ---------------- #
def query_llm(prompt: str, retries: int = 3, delay: int = 2) -> str:
    """
    Query Hugging Face Inference API with retry & error handling
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 60,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=15
            )

            # ---------------- HANDLE RESPONSE ---------------- #

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "").strip()

                return str(data)

            # 🔁 Model loading (common HF behavior)
            elif response.status_code == 503:
                print("⏳ Model loading... retrying")
                time.sleep(delay)

            # 🚫 Rate limit / auth issues
            else:
                return f"API Error {response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            print("⏳ Timeout... retrying")
            time.sleep(delay)

        except Exception as e:
            return f"LLM Exception: {str(e)}"

    return "⚠️ LLM failed after retries"