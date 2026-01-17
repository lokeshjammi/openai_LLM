import glob

import tiktoken

model = "gpt-4.1-mini"
db_name = "solar_vector_db"


knowledge_base_path = "solar_knowledge_base/**/*.md"
files = glob.glob(knowledge_base_path)

solar_entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r') as f:
        solar_entire_knowledge_base += f.read()
        solar_entire_knowledge_base += "\n\n"

encoding = tiktoken.encoding_for_model(model)
tokens = encoding.encode(solar_entire_knowledge_base)

print(tokens)