from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API Key loaded successfully.")
else:
    print("Failed to load API Key.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ],
    stream=True
)

# print(response.choices[0].message.content)
for chunk in response:
    if chunk.choices[0].message.content:
        print(chunk.choices[0].message.content, end="", flush=True)