import glob

from langchain_core import documents
import tiktoken
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.system('clear')
model = "gpt-4.1-mini"
db_name = "solar_vector_db"


knowledge_base_path = "solar_knowledge_base/**/*.md"
files = glob.glob(knowledge_base_path)

solar_entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r') as f:
        solar_entire_knowledge_base += f.read()
        solar_entire_knowledge_base += "\n\n"

# encoding = tiktoken.encoding_for_model(model)
# tokens = encoding.encode(solar_entire_knowledge_base)


folders = glob.glob("solar_knowledge_base/*")
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
print(vector_store)