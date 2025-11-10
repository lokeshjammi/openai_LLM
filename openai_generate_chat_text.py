from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr


load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully.\n")
else:
    print("❌ Failed to load API Key.")
    exit()

client = OpenAI(api_key=api_key)

def chat_with_model(messages, request=None):
    full_response = ""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, list):
        messages.insert(0, {"role": "system", "content": "You are a helpful AI tutor."})
    else:
        messages = [{"role": "system", "content": "You are a helpful AI tutor."}] + messages
    with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=messages
    ) as stream:
        for chunk in stream:
            if chunk.type == "content.delta":
                full_response += chunk.delta
                yield full_response

    return full_response


view = gr.ChatInterface(
    fn=chat_with_model,
    type="messages",
    title="💬 Chat with Your Friendly AI Tutor",
    description="A simple chat interface to interact with the GPT-4o-mini model."
)
view.launch()