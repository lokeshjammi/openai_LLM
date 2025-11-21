from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(".env")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGING_FACE_API_KEY"),
)
def learn_inference():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Explain the theory of relativity in simple terms.",
        },
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages
    )

    print(response.choices[0].message.content)

learn_inference()