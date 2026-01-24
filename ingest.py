import glob

from langchain_core import documents
import tiktoken
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.system('clear')
MODEL = "gpt-4.1-mini"
DB_NAME = "solar_vector_db"

def document_loader():
    folders = glob.glob("solar_knowledge_base/*")
    documents = []
    for folder in folders:
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        folder_docs = loader.load()
        for doc in folder_docs:
            documents.append(doc)
    return documents

def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embedding_model(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
    return vector_store

if __name__ == "__main__":
    documents = document_loader()
    chunks = text_splitter(documents)
    embedding_model(chunks)
    print("Ingest Completed")