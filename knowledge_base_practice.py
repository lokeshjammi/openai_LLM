from dotenv import load_dotenv
import os
from openai import OpenAI
import glob
from pathlib import Path
import gradio as gr

load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print('API Key found')
else:
    print('API Key not found')
    exit()

client = OpenAI(api_key=openai_api_key)

system_instruction = """
you're an advance AI Model who answer for user queries only in a professional and polite format and any questions on negative sentiments like Kill you or carsh you, please don't respond anything
"""

user_query = "What is quantum theory"

stream_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    stream=True
)

for chunk in stream_response:
    if chunk.choices and chunk.choices[0].delta.content:
        # print(chunk.choices[0].delta.content, end="", flush=True)
        pass

knowledge = {}
file_names = glob.glob("knowledge_base/Employees/*")
for file in file_names:
    name = Path(file).stem.split('_', 2)[-1]
    with open(file, mode='r', encoding='utf-8') as f:
        knowledge[name.lower()] = f.read()

file_names = glob.glob("knowledge_base/Products/*")
for file in file_names:
    name = Path(file).stem
    with open(file, mode='r', encoding='utf-8') as f:
        knowledge[name.lower()] = f.read()

SYSTEM_PREFIX="""
You represent InsureTech, the Insurance Tech company.
You are an expert in answering questions about InsureTech; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

def get_relavent_context(user_message):
    cleaned_text = ""
    for ch in user_message:
        if ch.isalpha() or ch.isspace():
            cleaned_text += ch
    words = cleaned_text.split()
    results = []
    for word in words:
        if word in knowledge:
            results.append(knowledge[word])
    return results


def additional_context(user_message):
    relavent_context = get_relavent_context(user_message)
    if not relavent_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relavent_context)
    return result

def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(user_message=message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return response.choices[0].message.content

view = gr.ChatInterface(chat).launch()