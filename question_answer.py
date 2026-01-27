from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


load_dotenv('.env')
openai_key = os.getenv('OPENAI_API_KEY')

if openai_key:
    print('API Key found')
else:
    print('API Key not found')
    exit()

MODEL = 'gpt-4.1-mini'
VECTOR_DB = str(Path(__file__).parent/'solar_vector_db')
K_RETREIVAL = 10

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})


vector_store = Chroma(persist_directory = VECTOR_DB, embedding_function=embedding)
retriever = vector_store.as_retriever()

llm = ChatOpenAI(model=MODEL, api_key=openai_key, temperature=0.2)

def fetch_context(query):
    context = retriever.invoke(query, k=K_RETREIVAL)
    print(context)
    SYSTEM_PROMPT = f"""
        You're a knowledgeable, friendly assistance who represent a solar technology company.
        You're chatting with an user about solar related products and other related information only.
        If the answer is known, generate the response using the context generated
        If you don't know the answer, say so.
        Context: {context}
    """
    System_prompt = SYSTEM_PROMPT.format(context)


fetch_context("How to install solar panels?")