from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_chroma import Chroma

model = "gpt-4.1-mini"
db_name = "vector_db"
load_dotenv('.env')

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print("API Key loaded")
else:
    print("API Key is not loaded")
    exit()

client = OpenAI(api_key=openai_api_key)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs = {"device": "cpu"})
vector_store = Chroma(persist_directory=db_name, embedding_function=embeddings)

