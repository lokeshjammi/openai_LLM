from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully.\n")
else:
    print("❌ Failed to load API Key.")
    exit()

client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

system_instruction = "You are an expert c++ developer, you can convert python code to c++ code."

def code_converter(python_code: str):
    prompt = (
        "You're a code converter, convert the following python code to c++ code:"
        "Give me the c++ code only, no other text or comments."
        f"{python_code}"
        "The response code should be saved in a file named 'code.cpp' and the file should be saved in the current directory."
    )

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    end_time = time.perf_counter()
    diff = end_time - start_time
    end_response = response.choices[0].message.content
    return end_response, diff

python_code = """
def add(a, b):
    return a + b
"""

c_code = code_converter(python_code)
print(c_code)
output_file = "code.cpp"

with open(output_file, "w") as file:
    file.write(c_code[0])

print(f"C++ code saved to {output_file}")