import requests
import os
from dotenv import load_dotenv

print("🚀 Starting HF Test...")

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

print("🔑 Token loaded:", HF_TOKEN is not None)

API_URL = "https://api-inference.huggingface.co/models/distilgpt2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

payload = {
    "inputs": "Explain CPU spike in one line:",
    "parameters": {
        "max_new_tokens": 40
    }
}

try:
    print("📡 Sending request...")

    response = requests.post(API_URL, headers=headers, json=payload)

    print("🔍 Status:", response.status_code)

    data = response.json()
    print("🔍 Raw:", data)

    if isinstance(data, list):
        print("✅ Output:", data[0]["generated_text"])

except Exception as e:
    print("❌ ERROR:", str(e))