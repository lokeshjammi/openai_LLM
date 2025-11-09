from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully.\n")
else:
    print("❌ Failed to load API Key.")
    exit()

client = OpenAI(api_key=api_key)

with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ]
) as stream:
    for chunk in stream:
        if chunk.type == "content.delta":
            print(chunk.delta, end="", flush=True)