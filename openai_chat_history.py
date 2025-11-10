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

def chat_with_model_with_history(query, history):
    if history == []:
        history.append(
            {
                "role": "system",
                "content": "You are a helpful AI tutor."
            }
        )
    user_query = []
    user_query.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history + user_query
    )

    ai_response = response.choices[0].message.content
    return ai_response

view = gr.ChatInterface(
    fn=chat_with_model_with_history,
    type="messages",
    title="💬 Chat with Your Friendly AI Tutor (with History)",
    description="A chat interface that maintains conversation history with the GPT-4o-mini model."
)
view.launch()