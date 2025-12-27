from dotenv import load_dotenv
import os
from openai import OpenAI
import glob
from pathlib import Path

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
    name = Path(file).stem.split('_', 1)[-1]
    with open(file, mode='r', encoding='utf-8') as f:
        knowledge[name.lower()] = f.read()

file_names = glob.glob("knowledge_base/Products/*")
for file in file_names:
    name = Path(file).stem
    with open(file, mode='r', encoding='utf-8') as f:
        knowledge[name.lower()] = f.read()

SYSTEM_PREFIX="""
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""