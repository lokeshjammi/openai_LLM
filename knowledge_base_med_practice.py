from dotenv import load_dotenv
import os
import glob
import tiktoken
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np

os.system('clear')
load_dotenv('.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("API Key loaded")
else:
    print("API Key is not loaded")

model = "gpt-4.1-mini"
db_name = "vector_db"

knowledge_base_path = "knowledge_base/**/*.md"
files = glob.glob(knowledge_base_path)

entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r') as f:
        entire_knowledge_base += f.read()
        entire_knowledge_base += "\n\n"

#Generate the token in all of the documents
encoding = tiktoken.encoding_for_model(model)
tokens = encoding.encode(entire_knowledge_base)

#Load in everything in the knowledge_base using Langchain's loader
folders = glob.glob("knowledge_base/*")
documents = []

for folder in folders:
    doc_type = os.path.basename(folder)
    # print(doc_type)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    # print(folder_docs)
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#Use a hugging face embeded model to encode the tokens
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
collections = vector_store._collection
sample_embedding = collections.get(include=["documents", "embeddings", "metadatas"])
vector = np.array(sample_embedding["embeddings"])
documents = sample_embedding['documents']
meta_datas = sample_embedding["metadatas"]