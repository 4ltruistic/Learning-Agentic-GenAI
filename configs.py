import os
os.environ["USER_AGENT"] = "my-rag-learning-app/1.0"
from langchain_community.document_loaders import  PyPDFDirectoryLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import  StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from tkinter import Tk, filedialog

chat_history=[]
PERSIST_DIR="./chroma_db"
Collection_name="research_papers"
context=[]


# Embeddings function
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generative model
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)

vector_db=Chroma(
        collection_name=Collection_name,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )