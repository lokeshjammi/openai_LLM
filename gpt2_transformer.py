# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("text-generation", model="openai-community/gpt2")
# print(pipe("Hello, how are you?")[0]['generated_text'])

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Explain the theory of relativity in simple terms.",
    }
]

prompt = tokenizer.encode()

print(prompt)