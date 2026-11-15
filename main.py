import os
import langchain
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loaded_docs=[]
docs_combined=[]

def docs_loader_list(loaded_doc):
    docs_combined.extend(loaded_doc)
    return loaded_doc



pdf_loader=PyPDFDirectoryLoader(
    path=r"D:\Research papers (DL)"
)

web_laoder=WebBaseLoader(
    web_path=r"https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"
)

web_docs=web_laoder.load()
pdf_docs=pdf_loader.load()
docs_loader_list(web_docs)
docs_loader_list(pdf_docs)

# Text Splitter

Text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
splitted_doc=Text_splitter.split_documents(docs_combined)

# Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector store

vector_db=Chroma.from_documents(
    documents=splitted_doc,
    embedding=embeddings,
    collection_name="research_papers",
    persist_directory="./chroma_db"
)

# retreiver
Retreiver=vector_db.as_retriever()



vecs=Retreiver.invoke("what is attention")

# for i, doc in enumerate(vecs):
#     print(f"\nResult {i+1}")
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)







