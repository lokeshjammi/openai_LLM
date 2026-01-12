from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

os.system('clear')
MODEL = "gpt-4.1-nano"
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

retriever = vector_store.as_retriever()
llm = ChatOpenAI(model=MODEL, temperature=0.2)

System_instruction = """
You are a helpful assistant who represent a company called insuretech that can answer questions about the documents in the knowledge base.
if relavent use the context to answer the question.
"""

def answer_question(query):
    docs = retriever.invoke(query)
    print(len(docs))
    context = ""
    for doc in docs:
        context += doc.page_content
    # print(context)
    system_prompt = System_instruction+"\n\nContext: "+context
    # print(system_prompt)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    # print(response.content)

answer_question("Who is carter?")