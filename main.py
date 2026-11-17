import os
os.environ["USER_AGENT"] = "my-rag-learning-app/1.0"
from langchain_community.document_loaders import  PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def docs_loader_list(*loaded_doc):
    combined=[]
    for docs in loaded_doc:
        combined.extend(docs)
    return combined


PERSIST_DIR="./chroma_db"
Collection_name="research_papers"
# Embeddings function
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db=Chroma(
        collection_name=Collection_name,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )
existing_data = vector_db.get()

if len(existing_data["ids"]) > 0:
    print("Fetching the embeddings locally!")

else:
    print("Creating the embeddings!")

    pdf_loader=PyPDFDirectoryLoader(
        path=r"D:\Research papers (DL)"
    )

    web_laoder=WebBaseLoader(
        web_path=r"https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"
    )

    web_docs=web_laoder.load()
    pdf_docs=pdf_loader.load()

    docs_combined=docs_loader_list(web_docs,pdf_docs)

    # Text Splitter

    Text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitted_doc=Text_splitter.split_documents(docs_combined)

    # vector store
    vector_db=Chroma.from_documents(
        documents=splitted_doc,
        embedding=embedding_model,
        collection_name=Collection_name,
        persist_directory="./chroma_db"
    )




# retreiver
Retreiver=vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k':2}
)


vecs=Retreiver.invoke("trump is gay")

for i, doc in enumerate(vecs):
    print(f"\nResult {i+1}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("=="*20)





